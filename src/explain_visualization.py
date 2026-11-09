"""Render aligned Grad-CAM maps and overlays for local academic review."""

from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from .explain import GradCAMResult
from .model import CLASSES


COLORMAP = "magma"
CATEGORY_NAMES = {
    "TP": "Verdadero positivo", "TN": "Verdadero negativo",
    "FP": "Falso positivo", "FN": "Falso negativo",
}


def _values(heatmap: torch.Tensor) -> np.ndarray:
    return heatmap.detach().cpu().nan_to_num(nan=0, posinf=0, neginf=0).clamp(0, 1).numpy()


def _colorize(values: np.ndarray) -> Image.Image:
    return Image.fromarray(colormaps[COLORMAP](values, bytes=True)).convert("RGB")


def make_overlay(image: Image.Image, heatmap: torch.Tensor, *, alpha: float = 0.55) -> Image.Image:
    """Blend the color map at native resolution, with zero opacity at zero signal."""
    if heatmap.shape != (image.height, image.width):
        raise ValueError("El mapa debe estar alineado con la imagen original.")
    if not 0 <= alpha <= 1:
        raise ValueError("La opacidad debe estar en [0, 1].")
    values = _values(heatmap)
    opacity = Image.fromarray(np.rint(255 * alpha * values).astype(np.uint8))
    return Image.composite(_colorize(values), image.convert("RGB"), opacity)


def _encode_png(image: Image.Image) -> bytes:
    with BytesIO() as stream:
        image.save(stream, format="PNG")
        return stream.getvalue()


def overlay_png(image: Image.Image, heatmap: torch.Tensor, *, alpha: float = 0.55) -> bytes:
    return _encode_png(make_overlay(image, heatmap, alpha=alpha))


def heatmap_png(heatmap: torch.Tensor) -> bytes:
    return _encode_png(_colorize(_values(heatmap)))


def save_figure(
    path: Path, image: Image.Image, result: GradCAMResult, *,
    image_id: str, true_label: int, category: str, alpha: float,
) -> None:
    figure = Figure(figsize=(13, 4.8), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    figure.suptitle(
        f"PAPILA · Validación · {image_id} · {CATEGORY_NAMES[category]} ({category})\n"
        f"Etiqueta real: {CLASSES[true_label]}   |   Predicción: {CLASSES[result.label]}   |   "
        f"Puntaje de glaucoma: {result.score_glaucoma:.4f}   |   Umbral: {result.threshold:.2f}",
        fontsize=13,
    )
    axes[0].imshow(image.convert("RGB"))
    heat = axes[1].imshow(result.heatmap, cmap=COLORMAP, vmin=0, vmax=1)
    axes[2].imshow(make_overlay(image, result.heatmap, alpha=alpha))
    for axis, title in zip(axes, ("Imagen original", "Grad-CAM · logit de glaucoma", "Superposición")):
        axis.set_title(title)
        axis.set_axis_off()
    figure.colorbar(heat, ax=axes, orientation="horizontal", shrink=0.30,
                    label="Intensidad relativa del mapa", location="bottom")
    figure.savefig(path, dpi=220)
