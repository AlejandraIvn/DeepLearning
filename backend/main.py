"""Local FastAPI service for the frozen academic PAPILA model."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from threading import Lock

import torch
from fastapi import FastAPI, HTTPException, Request, Response
from PIL import Image, ImageOps
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse

from src.device import select_device
from src.explain import TARGET_LAYER, explain_image
from src.explain_visualization import heatmap_png, overlay_png
from src.model import CLASSES, load_checkpoint
from src.predict import checkpoint_transform, score_image
from src.pretrained import PapilaResNet18


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "papila-internal-test.json"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 6_000_000
MAX_SIDE = 4096
PREVIEW_SIDE = 1024
UPLOAD_TIMEOUT = 15
MEDIA_TYPES = {"image/jpeg": "JPEG", "image/png": "PNG"}
NO_STORE = {"Cache-Control": "no-store"}
NOT_READY = "El modelo académico no está disponible. Revisa el servidor de análisis."
logger = logging.getLogger("uvicorn.error")


class Prediction(BaseModel):
    score_glaucoma: float
    label: int
    classification: str
    threshold: float


class Explanation(BaseModel):
    heatmap_png: str
    overlay_png: str
    target_layer: str
    has_signal: bool


class Analysis(Prediction, Explanation):
    pass


def _error(status: int, detail: str) -> HTTPException:
    return HTTPException(status, detail, headers=NO_STORE)


@dataclass
class ModelService:
    model: PapilaResNet18
    checkpoint: dict
    threshold: float
    device: torch.device
    lock: Lock = field(default_factory=Lock, repr=False)

    def process(self, payload: bytes, image_format: str, *, explain: bool) -> Prediction | Analysis:
        with self.lock, _decode_image(payload, image_format, self.checkpoint["architecture"]["image_size"]) as image:
            if not explain:
                score = score_image(self.model, image, checkpoint_transform(self.checkpoint), self.device)
                label = int(score >= self.threshold)
                return Prediction(score_glaucoma=score, label=label, classification=CLASSES[label],
                                  threshold=self.threshold)

            result = explain_image(self.model, image, self.checkpoint, threshold=self.threshold)
            image.thumbnail((PREVIEW_SIDE, PREVIEW_SIDE), Image.Resampling.BILINEAR)
            heatmap = result.heatmap
            if heatmap.shape != (image.height, image.width):
                heatmap = torch.nn.functional.interpolate(
                    heatmap[None, None], size=(image.height, image.width), mode="bilinear", align_corners=False,
                )[0, 0]
            return Analysis(
                score_glaucoma=result.score_glaucoma, label=result.label,
                classification=CLASSES[result.label], threshold=result.threshold,
                heatmap_png=base64.b64encode(heatmap_png(heatmap)).decode("ascii"),
                overlay_png=base64.b64encode(overlay_png(image, heatmap)).decode("ascii"),
                target_layer=TARGET_LAYER, has_signal=bool(heatmap.any()),
            )


def load_model() -> ModelService:
    spec = json.loads(CONFIG.read_text(encoding="utf-8"))["models"]["ResNet18"]
    # Load the same bytes whose digest was verified, even if the file is replaced.
    content = (ROOT / spec["checkpoint"]).read_bytes()
    if hashlib.sha256(content).hexdigest() != spec["sha256"]:
        raise ValueError("SHA-256 del checkpoint ResNet18 distinto del congelado.")
    device = select_device(prefer_cuda=os.environ.get("GLAUCOAI_CPU") != "1", enable_amp=False).device
    if device.type == "cuda":
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    with BytesIO(content) as stream:
        model, checkpoint = load_checkpoint(stream, device)
    if not isinstance(model, PapilaResNet18):
        raise ValueError("Se esperaba un checkpoint ResNet18.")
    model.requires_grad_(False)
    # Inference uses the frozen evaluation threshold, which is separate from training metadata.
    return ModelService(model, checkpoint, spec["threshold"], device)


def _decode_image(payload: bytes, image_format: str, image_size: int) -> Image.Image:
    try:
        with BytesIO(payload) as stream, Image.open(stream, formats=[image_format]) as source:
            width, height = source.size
            if max(width, height) > MAX_SIDE or width * height > MAX_PIXELS:
                raise _error(413, "La imagen supera 6 megapíxeles o 4096 píxeles por lado.")
            if getattr(source, "n_frames", 1) != 1 or min(width, height) * image_size < max(width, height):
                raise _error(422, "Usa una imagen fija de fondo de ojo con proporciones válidas.")
            oriented = ImageOps.exif_transpose(source)
            if "A" in oriented.getbands() or "transparency" in oriented.info:
                rgba = oriented.convert("RGBA")
                background = Image.new("RGB", rgba.size)
                background.paste(rgba, mask=rgba.getchannel("A"))
                return background
            return oriented.convert("RGB")
    except Image.DecompressionBombError:
        raise _error(413, "La imagen supera el límite de píxeles.") from None
    except (OSError, ValueError, SyntaxError):
        raise _error(422, "No se pudo decodificar la imagen como JPG o PNG.") from None


async def _read_image(request: Request) -> tuple[bytes, str]:
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type not in MEDIA_TYPES:
        raise _error(415, "Envía una imagen JPG o PNG.")
    size = request.headers.get("content-length")
    if size is not None and size.isdecimal() and int(size) > MAX_UPLOAD_BYTES:
        raise _error(413, "La imagen supera el límite de 10 MiB.")

    buffer = bytearray()
    try:
        async with asyncio.timeout(UPLOAD_TIMEOUT):
            async for chunk in request.stream():
                if len(buffer) + len(chunk) > MAX_UPLOAD_BYTES:
                    raise _error(413, "La imagen supera el límite de 10 MiB.")
                buffer.extend(chunk)
    except TimeoutError:
        raise _error(408, "La carga de la imagen tardó demasiado.") from None
    except ClientDisconnect:
        raise _error(400, "La carga de la imagen se interrumpió.") from None
    if not buffer:
        raise _error(422, "Selecciona una imagen antes de analizar.")
    return bytes(buffer), MEDIA_TYPES[media_type]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = None
    app.state.slots = asyncio.Semaphore(2)
    try:
        app.state.service = load_model()
    except FileNotFoundError:
        logger.error("Falta el checkpoint congelado o su configuración en el repositorio.")
    except ValueError as error:
        logger.error("No se pudo verificar el modelo: %s", error)
    except (OSError, RuntimeError, KeyError):
        logger.error("No se pudo cargar el modelo académico.")
    try:
        yield
    finally:
        app.state.service = None


app = FastAPI(title="GlaucoAI · API académica", lifespan=lifespan)


@app.get("/health")
async def health(request: Request):
    service = request.app.state.service
    if service is None:
        return JSONResponse({"status": "unavailable", "model_ready": False}, status_code=503, headers=NO_STORE)
    return {"status": "ok", "model_ready": True, "model": "resnet18", "device": str(service.device),
            "threshold": service.threshold}


async def _process(request: Request, *, explain: bool):
    service = request.app.state.service
    if service is None:
        raise _error(503, NOT_READY)
    async with request.app.state.slots:
        payload, image_format = await _read_image(request)
        try:
            result = await run_in_threadpool(service.process, payload, image_format, explain=explain)
        except (RuntimeError, OSError, ValueError):
            logger.error("Falló el procesamiento de una solicitud de análisis.")
            raise _error(500, "El servidor no pudo procesar la imagen.") from None
    return result


UPLOAD_SCHEMA = {"requestBody": {"required": True, "content": {
    media_type: {"schema": {"type": "string", "format": "binary"}} for media_type in MEDIA_TYPES
}}}


@app.post("/predict", response_model=Prediction, openapi_extra=UPLOAD_SCHEMA)
async def predict(request: Request, response: Response):
    response.headers.update(NO_STORE)
    return await _process(request, explain=False)


@app.post("/explain", response_model=Explanation, openapi_extra=UPLOAD_SCHEMA)
async def explain(request: Request, response: Response):
    response.headers.update(NO_STORE)
    return await _process(request, explain=True)


@app.post("/analyze", response_model=Analysis, openapi_extra=UPLOAD_SCHEMA)
async def analyze(request: Request, response: Response):
    response.headers.update(NO_STORE)
    return await _process(request, explain=True)
