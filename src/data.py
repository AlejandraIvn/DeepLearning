"""Load a frozen PAPILA manifest without regenerating its patient splits."""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .device import DeviceConfig, select_device


SPLITS = ("train", "val", "test")
MANIFEST_COLUMNS = {"image_id", "patient_id", "eye", "label", "relative_path", "split"}
Sample = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    patient_id: str
    eye: str
    label: int
    relative_path: PurePosixPath


@dataclass(frozen=True)
class Normalization:
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


def make_image_transform(
    image_size: int, normalization: Normalization | None = None, *, augment: bool = False,
) -> transforms.Compose:
    """Share the RGB image preprocessing across training, evaluation and inference."""
    steps = [
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ]
    if augment:
        steps.append(transforms.RandomHorizontalFlip(p=0.5))
    if normalization is not None:
        steps.append(transforms.Normalize(normalization.mean, normalization.std))
    return transforms.Compose(steps)


class PapilaDataset(Dataset[Sample]):
    """Decode and transform one RGB image at a time on the CPU."""

    def __init__(
        self,
        dataset_dir: Path,
        records: Sequence[ImageRecord],
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self.dataset_dir = dataset_dir
        self.records = tuple(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Sample:
        record = self.records[index]
        path = self.dataset_dir / record.relative_path
        try:
            with Image.open(path) as image:
                pixels = self.transform(image if image.mode == "RGB" else image.convert("RGB"))
        except (OSError, Image.DecompressionBombError) as error:
            raise ValueError(f"No se pudo cargar {record.image_id} desde {path}: {error}") from error
        return pixels, torch.tensor(record.label, dtype=torch.float32)


@dataclass(frozen=True)
class PapilaLoaders:
    train: DataLoader[Sample]
    val: DataLoader[Sample]
    test: DataLoader[Sample] | None
    normalization: Normalization


def _read_manifest(manifest: Path) -> dict[str, list[ImageRecord]]:
    records: dict[str, list[ImageRecord]] = {split: [] for split in SPLITS}
    patient_splits: dict[str, str] = {}
    image_ids: set[str] = set()
    eyes: set[tuple[str, str]] = set()
    paths: set[Path] = set()
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = MANIFEST_COLUMNS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Columnas ausentes en {manifest}: {', '.join(sorted(missing))}.")
        for row in reader:
            location = f"{manifest}, línea {reader.line_num}"
            if None in row or any(not row[column] for column in MANIFEST_COLUMNS):
                raise ValueError(f"Fila incompleta o con campos extra en {location}.")
            image_id, patient, eye = row["image_id"], row["patient_id"], row["eye"]
            split, label = row["split"], row["label"]
            if split not in records or label not in {"0", "1"} or eye not in {"OD", "OS"}:
                raise ValueError(f"Split, etiqueta u ojo inválido en {location}.")
            raw_path = row["relative_path"]
            path = PurePosixPath(raw_path)
            if (
                not path.parts or path.is_absolute() or ".." in path.parts
                or "\\" in raw_path or PureWindowsPath(raw_path).drive
            ):
                raise ValueError(f"Ruta relativa inválida en {location}: {raw_path!r}.")
            previous_split = patient_splits.setdefault(patient, split)
            if previous_split != split:
                raise ValueError(f"Paciente {patient} aparece en {previous_split} y {split} ({location}).")
            if image_id in image_ids or (patient, eye) in eyes or Path(path) in paths:
                raise ValueError(f"Imagen, ojo o ruta duplicada para {image_id} en {location}.")
            image_ids.add(image_id)
            eyes.add((patient, eye))
            paths.add(Path(path))
            records[split].append(ImageRecord(image_id, patient, eye, int(label), path))
    for split, rows in records.items():
        if not rows:
            raise ValueError(f"El manifiesto {manifest} no contiene filas para {split}.")
    return records


def _fit_normalization(train: PapilaDataset) -> Normalization:
    """Accumulate per-channel population statistics over resized training pixels."""
    pixel_sum = torch.zeros(3, dtype=torch.float64)
    square_sum = torch.zeros_like(pixel_sum)
    pixels = 0
    for index in range(len(train)):
        image, _ = train[index]
        image = image.to(dtype=torch.float64)
        pixel_sum += image.sum(dim=(1, 2))
        square_sum += image.square().sum(dim=(1, 2))
        pixels += image.shape[1] * image.shape[2]
    mean = pixel_sum / pixels
    std = (square_sum / pixels - mean.square()).clamp_min(0).sqrt()
    std[std <= torch.finfo(torch.float32).eps] = 1.0
    return Normalization(
        mean=(mean[0].item(), mean[1].item(), mean[2].item()),
        std=(std[0].item(), std[1].item(), std[2].item()),
    )


def make_loaders(
    dataset_dir: Path,
    manifest: Path,
    *,
    device: DeviceConfig | None = None,
    batch_size: int = 16,
    num_workers: int = 0,
    image_size: int = 96,
    seed: int = 42,
    augment_train: bool = True,
    include_test: bool = False,
) -> PapilaLoaders:
    """Build CPU loaders using the CSV assignments. Enable test only for final evaluation."""
    if batch_size < 1 or num_workers < 0 or image_size < 1:
        raise ValueError("batch_size e image_size deben ser positivos y num_workers no puede ser negativo.")
    records = _read_manifest(manifest)
    device = device if device is not None else select_device()
    base = make_image_transform(image_size)
    stats = _fit_normalization(PapilaDataset(dataset_dir, records["train"], base))
    evaluate = make_image_transform(image_size, stats)
    train_transform = make_image_transform(image_size, stats, augment=True) if augment_train else evaluate

    def loader(split: str, transform: Callable[[Image.Image], torch.Tensor]) -> DataLoader[Sample]:
        return DataLoader(
            PapilaDataset(dataset_dir, records[split], transform),
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=device.pin_memory,
            generator=torch.Generator().manual_seed(seed),
        )

    return PapilaLoaders(
        train=loader("train", train_transform),
        val=loader("val", evaluate),
        test=loader("test", evaluate) if include_test else None,
        normalization=stats,
    )
