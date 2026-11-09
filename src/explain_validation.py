"""Generate Grad-CAM figures and a benchmark from frozen PAPILA validation only."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch
from PIL import Image

from .data import ImageRecord, PapilaDataset, _read_manifest
from .explain import TARGET_LAYER, explain_image, summarize_regions
from .explain_visualization import CATEGORY_NAMES, save_figure
from .internal_test import (
    CONFIG_PATH, ROOT, _forward, _state_hash, checkpoint_metadata, cohort_identity, sha256_file,
)
from .model import load_checkpoint
from .predict import checkpoint_transform


SELECTION_RULE = "Primer image_id en orden lexicográfico por categoría, con puntajes FP32 de CPU."


def select_examples(
    scored: Iterable[tuple[ImageRecord, float]], threshold: float,
) -> dict[str, tuple[ImageRecord, float]]:
    categories = {(1, 1): "TP", (0, 0): "TN", (0, 1): "FP", (1, 0): "FN"}
    selected = {}
    for record, score in sorted(scored, key=lambda row: row[0].image_id):
        if not math.isfinite(score):
            raise ValueError(f"Puntaje no finito para {record.image_id}.")
        category = categories[record.label, int(score >= threshold)]
        selected.setdefault(category, (record, score))
    return selected


def _spatial_summary(image: Image.Image, heatmap: torch.Tensor) -> dict:
    height, width = heatmap.shape
    border_y, border_x = round(height * 0.1), round(width * 0.1)
    border = torch.ones_like(heatmap, dtype=torch.bool)
    border[border_y:height - border_y, border_x:width - border_x] = False
    dark = torch.from_numpy(np.asarray(image).max(axis=2) <= 20)
    return summarize_regions(heatmap, {"outer_10_percent": border, "dark_rgb_max_le_20": dark})


def run_validation(dataset_dir: Path, manifest: Path, output: Path, *, cpu: bool, alpha: float) -> dict:
    if not 0 <= alpha <= 1:
        raise ValueError("La opacidad debe estar en [0, 1].")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    spec = config["models"]["ResNet18"]
    path = ROOT / spec["checkpoint"]
    reference = ROOT / config["models"]["CNN"]["checkpoint"]
    threshold = spec["threshold"]  # The checkpoint's training threshold is different.
    files = {"resnet18": path, "cnn_reference": reference, "manifest": manifest}
    expected = {"resnet18": spec["sha256"], "cnn_reference": config["models"]["CNN"]["sha256"],
                "manifest": config["dataset"]["manifest_sha256"]}
    # The historical CNN is optional for Grad-CAM, but verify it if present.
    if not reference.is_file():
        del files["cnn_reference"]
    before = {name: sha256_file(file) for name, file in files.items()}
    for name, digest in before.items():
        if digest != expected[name]:
            raise ValueError(f"SHA-256 distinto del congelado para {name}.")

    records = _read_manifest(manifest)["val"]
    if cohort_identity(records, dataset_dir) != config["dataset"]["splits"]["val"]:
        raise ValueError("La identidad de validación difiere de la configuración congelada.")
    model, checkpoint = load_checkpoint(path)
    metadata = json.loads(json.dumps(checkpoint_metadata(checkpoint)))
    if metadata != spec["metadata"] or checkpoint["manifest_sha256"] != expected["manifest"]:
        raise ValueError("El checkpoint difiere de la arquitectura o procedencia congelada.")
    model.requires_grad_(False)
    state_before = _state_hash(model)
    transform = checkpoint_transform(checkpoint)
    dataset = PapilaDataset(dataset_dir, records, transform)
    scored = [(record, _forward(model, dataset[i][0].unsqueeze(0)).sigmoid().item())
              for i, record in enumerate(records)]
    selected = select_examples(scored, threshold)
    devices = [torch.device("cpu")]
    if not cpu and torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))

    output.mkdir(parents=True, exist_ok=False)
    examples = {}
    images = {}
    for category in CATEGORY_NAMES:
        if category not in selected:
            continue
        record, selection_score = selected[category]
        with Image.open(dataset_dir / record.relative_path) as source:
            image = source.convert("RGB")
        images[category] = image
        examples[category] = {
            "image_id": record.image_id, "patient_id": record.patient_id, "eye": record.eye,
            "true_label": record.label, "label": int(selection_score >= threshold),
            "category": category, "selection_score_cpu": selection_score,
            "original_size": image.size, "devices": {},
        }

    benchmarks = {}
    cpu_maps = {}
    for device in devices:
        model.to(device)
        use_cuda = device.type == "cuda"
        warmup = Image.new("RGB", next(iter(examples.values()))["original_size"])
        explain_image(model, warmup, checkpoint, threshold=threshold)
        elapsed = []
        peaks = []
        for category, example in examples.items():
            record, selection_score = selected[category]
            image = images[category]
            baseline = _forward(model, transform(image).unsqueeze(0).to(device)).sigmoid().item()
            if use_cuda:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                allocated_before = torch.cuda.memory_allocated(device)
            start = perf_counter()
            result = explain_image(model, image, checkpoint, threshold=threshold)
            if use_cuda:
                torch.cuda.synchronize(device)
            elapsed.append((perf_counter() - start) * 1000)

            if not math.isclose(result.score_glaucoma, baseline, rel_tol=1e-5, abs_tol=1e-6):
                raise ValueError(f"El puntaje de {record.image_id} cambió con Grad-CAM en {device}.")
            if result.label != example["label"] or not math.isclose(
                result.score_glaucoma, selection_score, rel_tol=1e-5, abs_tol=1e-6,
            ):
                raise ValueError(f"El puntaje o la clase de {record.image_id} no coincide con la selección CPU.")
            stats = {"score_glaucoma": result.score_glaucoma, "score_without_grad_cam": baseline,
                     "grad_cam_ms": elapsed[-1]}
            if use_cuda:
                peak = torch.cuda.max_memory_allocated(device)
                peaks.append(peak)
                stats.update({"peak_allocated_bytes": peak, "extra_allocated_bytes": peak - allocated_before})
                difference = (result.heatmap - cpu_maps[category]).abs()
                example["cpu_cuda_map_difference"] = {"mean_absolute": difference.mean().item(),
                                                      "max_absolute": difference.max().item()}
            else:
                cpu_maps[category] = result.heatmap
            example["devices"][str(device)] = stats

            if device == devices[-1]:
                stem = f"{category}-{record.image_id}"
                save_figure(output / f"{stem}.png", image, result, image_id=record.image_id,
                            true_label=record.label, category=category, alpha=alpha)
                np.save(output / f"{stem}.npy", result.heatmap.numpy(), allow_pickle=False)
                example.update({"figure": f"{stem}.png", "heatmap": f"{stem}.npy",
                                "feature_shape": result.feature_shape, "content_box": result.content_box,
                                "spatial_summary": _spatial_summary(image, result.heatmap)})

        benchmarks[str(device)] = {
            "median_grad_cam_ms": median(elapsed), "max_grad_cam_ms": max(elapsed),
            "peak_allocated_bytes": max(peaks) if peaks else None,
            "device_name": torch.cuda.get_device_name(device) if use_cuda else "CPU",
        }
        if _state_hash(model) != state_before:
            raise ValueError(f"Los pesos o buffers cambiaron durante el análisis en {device}.")

    after = {name: sha256_file(file) for name, file in files.items()}
    if after != before:
        raise ValueError("Un checkpoint o el manifiesto cambió durante el análisis.")
    report = {
        "split": "val", "validation_images": len(records), "selection_rule": SELECTION_RULE,
        "threshold": threshold, "target_layer": TARGET_LAYER, "target": "logit de glaucoma",
        "dtype": "float32", "torch": str(torch.__version__), "cpu_threads": torch.get_num_threads(),
        "cuda_version": torch.version.cuda if len(devices) > 1 else None,
        "overlay_alpha": alpha, "figure_device": str(devices[-1]),
        "missing_categories": [category for category in CATEGORY_NAMES if category not in selected],
        "examples": examples, "benchmarks": benchmarks, "model_state_unchanged": True,
        "sha256_before": before, "sha256_after": after,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true", help="Omitir la comprobación CUDA.")
    parser.add_argument("--alpha", type=float, default=0.55, help="Opacidad máxima de la superposición.")
    args = parser.parse_args()
    output = ROOT / "models" / "runs" / ("papila-grad-cam-cpu" if args.cpu else "papila-grad-cam")
    if not args.cpu and torch.cuda.is_available():
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    try:
        run_validation(args.dataset_dir, args.manifest, output, cpu=args.cpu, alpha=args.alpha)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.error(str(error))
    print(output / "report.json")


if __name__ == "__main__":
    main()
