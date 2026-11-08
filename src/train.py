"""Train the PAPILA CNN using training and validation data only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from .data import Sample, make_loaders
from .device import DeviceConfig, select_device
from .model import ARCHITECTURE, CHECKPOINT_VERSION, CLASSES, PapilaCNN, read_checkpoint


@dataclass(frozen=True)
class TrainConfig:
    image_size: int = 96
    batch_size: int = 16
    num_workers: int = 0
    learning_rate: float = 0.001
    epochs: int = 40
    patience: int = 5
    seed: int = 42
    weight_decay: float = 0.0
    dropout: float = 0.0
    augment_train: bool = True
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if min(self.epochs, self.patience, self.batch_size) < 1 or self.image_size < 4 or self.num_workers < 0:
            raise ValueError("Épocas, paciencia y batch deben ser positivos, resolución ≥ 4 y workers ≥ 0.")
        if not (0 <= self.seed < 2**32 and 0 <= self.dropout <= 1 and 0 <= self.threshold <= 1):
            raise ValueError("Semilla fuera de rango o dropout/umbral fuera de [0, 1].")
        if not (math.isfinite(self.learning_rate) and self.learning_rate > 0
                and math.isfinite(self.weight_decay) and self.weight_decay >= 0):
            raise ValueError("Learning rate debe ser positivo y weight decay no negativo, ambos finitos.")


@dataclass
class EarlyStopping:
    patience: int
    best_loss: float = math.inf
    stale_epochs: int = 0

    def step(self, loss: float) -> tuple[bool, bool]:
        """Report strict improvement and whether consecutive stalls exhaust patience."""
        improved = loss < self.best_loss
        if improved:
            self.best_loss = loss
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        return improved, self.stale_epochs >= self.patience


def _seed_run(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _train_epoch(
    model: PapilaCNN, loader: DataLoader[Sample], device: DeviceConfig,
    criterion: nn.BCEWithLogitsLoss, optimizer: torch.optim.Adam, scaler: torch.amp.GradScaler,
) -> tuple[float, int]:
    model.train()
    loss_sum = 0.0
    samples = 0
    skipped = 0
    for batch_number, batch in enumerate(loader, start=1):
        images, labels = device.move_batch(batch)
        optimizer.zero_grad(set_to_none=True)
        with device.autocast():
            loss = criterion(model(images), labels)
        value = loss.item()
        if not math.isfinite(value):
            raise FloatingPointError(f"Pérdida de entrenamiento no finita en lote {batch_number}: {value}.")
        scaler.scale(loss).backward()
        if not scaler.is_enabled():
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            if not gradients or not torch.stack([gradient.isfinite().all() for gradient in gradients]).all().item():
                raise FloatingPointError(f"Gradientes ausentes o no finitos en lote {batch_number}.")
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped += int(scaler.get_scale() < previous_scale)
        loss_sum += value * labels.numel()
        samples += labels.numel()
    if skipped == len(loader):
        raise FloatingPointError("GradScaler omitió todas las actualizaciones de la época.")
    return loss_sum / samples, skipped


def validation_metrics(labels: torch.Tensor, logits: torch.Tensor, threshold: float) -> dict:
    """Use glaucoma as positive. PR-AUC is trapezoidal area under the PR curve."""
    predicted = torch.sigmoid(logits) >= threshold
    tn, fp, fn, tp = (int(value) for value in confusion_matrix(labels, predicted, labels=[0, 1]).ravel())
    roc_auc = pr_auc = None
    if tn + fp and tp + fn:
        roc_auc = float(roc_auc_score(labels, logits))
        precision_curve, recall_curve, _ = precision_recall_curve(labels, logits)
        pr_auc = float(auc(recall_curve, precision_curve))
    return {
        "val_accuracy": (tn + tp) / labels.numel(),
        "val_precision": tp / (tp + fp) if tp + fp else 0.0,
        "val_recall": tp / (tp + fn) if tp + fn else 0.0,
        "val_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "val_roc_auc": roc_auc, "val_pr_auc": pr_auc,
        "val_tn": tn, "val_fp": fp, "val_fn": fn, "val_tp": tp,
    }


@torch.inference_mode()
def _validate(
    model: PapilaCNN, loader: DataLoader[Sample], device: DeviceConfig,
    criterion: nn.BCEWithLogitsLoss, threshold: float,
) -> dict:
    model.eval()
    scores, targets = [], []
    loss_sum = 0.0
    for batch_number, batch in enumerate(loader, start=1):
        images, labels = device.move_batch(batch)
        logits = model(images)  # Match FP32 inference for checkpoint comparison.
        loss = criterion(logits, labels).item()
        if not math.isfinite(loss):
            raise FloatingPointError(f"Pérdida de validación no finita en lote {batch_number}: {loss}.")
        loss_sum += loss * labels.numel()
        scores.append(logits.cpu())
        targets.append(labels.cpu())
    labels = torch.cat(targets)
    return {"val_loss": loss_sum / labels.numel(),
            **validation_metrics(labels, torch.cat(scores), threshold)}


def _save_checkpoint(path: Path, payload: dict) -> None:
    handle = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            torch.save(payload, handle)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _plot_loss(history: list[dict], best_epoch: int, output: Path) -> None:
    figure = Figure(figsize=(8, 4.5), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    epochs = [row["epoch"] for row in history]
    axis.plot(epochs, [row["train_loss"] for row in history], marker=".", label="Entrenamiento")
    axis.plot(epochs, [row["val_loss"] for row in history], marker=".", label="Validación")
    axis.axvline(best_epoch, linestyle="--", color="gray", label=f"Época seleccionada ({best_epoch})")
    axis.set(title="CNN de PAPILA · pérdida por época", xlabel="Época", ylabel="BCE con logits")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.savefig(output, dpi=150)


def run_training(
    dataset_dir: Path, manifest: Path, run_dir: Path, config: TrainConfig,
    device: DeviceConfig | None = None,
) -> dict:
    """Fit from scratch, checkpoint on validation loss, and restore the selected weights."""
    if run_dir.exists():
        raise FileExistsError(f"La carpeta de la corrida ya existe: {run_dir}. Usa otra ruta.")
    _seed_run(config.seed)
    device = device if device is not None else select_device()
    use_cuda = device.device.type == "cuda"
    if use_cuda:
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device.device)
    started = time.perf_counter()
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    loaders = make_loaders(
        dataset_dir, manifest, device=device, batch_size=config.batch_size,
        num_workers=config.num_workers, image_size=config.image_size, seed=config.seed,
        augment_train=config.augment_train, include_test=False,
    )
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != digest:
        raise ValueError("El manifiesto cambió durante la carga.")

    run_dir.mkdir(parents=True)
    checkpoint_path = run_dir / "best.pt"
    model = PapilaCNN(config.image_size, config.dropout).to(device.device)
    initial_weight = model.fc_out.weight.detach().clone()
    initial_bias = model.fc_out.bias.detach().clone()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = device.make_grad_scaler()
    stopping = EarlyStopping(config.patience)
    history: list[dict] = []
    metadata = {
        "format_version": CHECKPOINT_VERSION,
        "architecture": {"name": ARCHITECTURE, "image_size": config.image_size, "dropout": config.dropout},
        "config": asdict(config), "optimizer": "Adam", "seed": config.seed,
        "normalization": asdict(loaders.normalization), "manifest_sha256": digest,
        "classes": CLASSES, "torch_version": str(torch.__version__),
        "preprocessing": {"color_mode": "RGB", "interpolation": "bilinear", "pixel_range": [0, 1]},
        "device": str(device.device), "amp_fp16": device.amp_enabled,
    }
    with (run_dir / "history.csv").open("w", encoding="utf-8", newline="") as stream:
        for epoch in range(1, config.epochs + 1):
            try:
                train_loss, skipped = _train_epoch(model, loaders.train, device, criterion, optimizer, scaler)
                validation = _validate(model, loaders.val, device, criterion, config.threshold)
            except FloatingPointError as error:
                raise FloatingPointError(f"Época {epoch}: {error}") from error
            row = {"epoch": epoch, "train_loss": train_loss, **validation, "amp_skipped_steps": skipped}
            if not history:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            history.append(row)
            writer.writerow(row)
            stream.flush()
            print(json.dumps(row, allow_nan=False), flush=True)
            improved, should_stop = stopping.step(validation["val_loss"])
            if improved:
                _save_checkpoint(checkpoint_path, {
                    **metadata, "epoch": epoch, "best_val_loss": stopping.best_loss, "validation": validation,
                    "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
                })
            if should_stop:
                break

    selected = read_checkpoint(checkpoint_path)
    model.load_state_dict(selected["state_dict"])
    if torch.equal(model.fc_out.weight, initial_weight) and torch.equal(model.fc_out.bias, initial_bias):
        raise FloatingPointError("Los pesos de salida restaurados no cambiaron respecto a la inicialización.")
    _plot_loss(history, selected["epoch"], run_dir / "loss.png")
    summary = {
        **metadata, "checkpoint": str(checkpoint_path), "epochs_run": len(history),
        "best_epoch": selected["epoch"], "best_val_loss": selected["best_val_loss"],
        "validation": selected["validation"], "early_stopped": should_stop,
        "train_images": len(loaders.train.dataset), "val_images": len(loaders.val.dataset),
        "amp_skipped_steps": sum(row["amp_skipped_steps"] for row in history), "updated_head": True,
        "gpu": torch.cuda.get_device_name(device.device) if use_cuda else None,
        "cuda_version": torch.version.cuda if use_cuda else None,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device.device) / 2**20 if use_cuda else None,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(device.device) / 2**20 if use_cuda else None,
        "seconds": time.perf_counter() - started,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=Path("models/runs/papila-reference"))
    for name, default in asdict(defaults).items():
        parser.add_argument(
            f"--{name.replace('_', '-')}", default=default,
            **({"action": argparse.BooleanOptionalAction} if isinstance(default, bool) else {"type": type(default)}),
        )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    try:
        config = TrainConfig(**{name: getattr(args, name) for name in asdict(defaults)})
        # Set the cuBLAS configuration before selecting/initializing CUDA.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        device = select_device(prefer_cuda=not args.cpu, enable_amp=not args.no_amp)
        summary = run_training(args.dataset_dir, args.manifest, args.run_dir, config, device)
    except torch.OutOfMemoryError as error:
        parser.exit(1, f"Memoria CUDA insuficiente. Reduce --batch-size y usa otra --run-dir. {error}\n")
    except (FloatingPointError, OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Entrenamiento interrumpido: {error}\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
