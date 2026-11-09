"""Validate a local PAPILA release and write a patient-grouped CSV manifest."""

from __future__ import annotations

import argparse
import csv
import random
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile

from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException
from PIL import Image


DEFAULT_SEED = 42
SPLIT_PERCENTAGES = {"train": 65, "val": 15, "test": 20}
IMAGE_NAME = re.compile(r"RET([0-9]+)(OD|OS)", re.IGNORECASE)
CLINICAL_ID = re.compile(r"#?([0-9]+)\s*(OD|OS)?", re.IGNORECASE)
EyeKey = tuple[str, str]


class PapilaError(ValueError):
    """An input file or record violates the PAPILA data contract."""


class ManifestRow(TypedDict):
    image_id: str
    patient_id: str
    eye: str
    label: int
    relative_path: str
    split: str


@dataclass(frozen=True)
class ClinicalLabel:
    diagnosis: int
    location: str


@dataclass(frozen=True)
class PreparationResult:
    rows: list[ManifestRow]
    excluded_suspects: int
    redundant_labels: int


def _patient_id(digits: str) -> str:
    return (digits.lstrip("0") or "0").zfill(3)


def _image_id(key: EyeKey) -> str:
    return f"RET{key[0]}{key[1]}"


def _index_images(directory: Path) -> dict[EyeKey, Path]:
    if not directory.is_dir():
        raise PapilaError(f"No existe el directorio de imágenes: {directory}")
    images: dict[EyeKey, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        match = IMAGE_NAME.fullmatch(path.stem)
        if match is None:
            raise PapilaError(f"Nombre de imagen inválido: {path.name}; se espera RET<id>OD/OS.")
        key = (_patient_id(match[1]), match[2].upper())
        if key in images:
            raise PapilaError(f"Imagen duplicada: {images[key].name} y {path.name} identifican {_image_id(key)}.")
        images[key] = path
    if not images:
        raise PapilaError(f"No se encontraron imágenes JPEG en {directory}.")
    return images


def _read_workbook(path: Path, default_eye: str) -> dict[EyeKey, ClinicalLabel]:
    if not path.is_file():
        raise PapilaError(f"No existe la planilla de etiquetas: {path}")
    labels: dict[EyeKey, ClinicalLabel] = {}
    diagnosis_column: int | None = None
    try:
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            sheet = workbook.worksheets[0]
            for number, row in enumerate(sheet.iter_rows(values_only=True), start=1):
                if not row or all(value is None for value in row):
                    continue
                location = f"{path.name}/{sheet.title}, fila {number}"
                identifier = str(row[0]).strip() if row[0] is not None else ""
                if not labels and identifier.upper() in {"", "ID"}:
                    headers = [
                        i for i, value in enumerate(row)
                        if isinstance(value, str) and value.strip().casefold() == "diagnosis"
                    ]
                    if headers:
                        if len(headers) != 1 or diagnosis_column not in {None, headers[0]}:
                            raise PapilaError(f"Columna Diagnosis ambigua en {location}.")
                        diagnosis_column = headers[0]
                        continue
                    # PAPILA repite el encabezado y tiene una fila separada con 'ID'.
                    if diagnosis_column is None or (identifier.upper() == "ID" and row[diagnosis_column] is None):
                        continue
                if diagnosis_column is None:
                    raise PapilaError(f"Falta la columna Diagnosis antes de los datos en {location}.")
                match = CLINICAL_ID.fullmatch(identifier)
                if match is None:
                    raise PapilaError(f"Identificador clínico inválido en {location}: {identifier!r}.")
                # El ojo explícito manda. El nombre de archivo solo completa IDs sin lateralidad.
                key = (_patient_id(match[1]), (match[2] or default_eye).upper())
                raw_diagnosis = row[diagnosis_column]
                if isinstance(raw_diagnosis, bool) or raw_diagnosis not in (0, 1, 2, "0", "1", "2"):
                    raise PapilaError(f"Diagnóstico inválido para {_image_id(key)} en {location}: "
                                      f"{raw_diagnosis!r}; se espera 0, 1 o 2.")
                if key in labels:
                    raise PapilaError(f"Etiqueta duplicada para {_image_id(key)} en {location}; "
                                      f"primera aparición en {labels[key].location}.")
                labels[key] = ClinicalLabel(int(raw_diagnosis), location)
        finally:
            workbook.close()
    except (BadZipFile, InvalidFileException, OSError, ParseError, IndexError, KeyError) as error:
        raise PapilaError(f"No se pudo leer la planilla {path}: {error}") from error
    if not labels:
        raise PapilaError(f"La planilla {path} no contiene registros clínicos válidos con Diagnosis.")
    return labels


def _read_labels(directory: Path) -> tuple[dict[EyeKey, ClinicalLabel], int]:
    labels = _read_workbook(directory / "patient_data_od.xlsx", "OD")
    redundant = 0
    for key, label in _read_workbook(directory / "patient_data_os.xlsx", "OS").items():
        previous = labels.get(key)
        if previous is None:
            labels[key] = label
        elif previous.diagnosis == label.diagnosis:
            redundant += 1
        else:
            raise PapilaError(f"Etiquetas contradictorias para {_image_id(key)}: "
                              f"{previous.diagnosis} en {previous.location} y "
                              f"{label.diagnosis} en {label.location}.")
    return labels, redundant


def _allocate_splits(size: int) -> dict[str, int]:
    """Round quotas within a profile by largest remainder, in split order on ties."""
    quotas = {split: divmod(size * pct, 100) for split, pct in SPLIT_PERCENTAGES.items()}
    counts = {split: whole for split, (whole, _) in quotas.items()}
    by_remainder = sorted(quotas, key=lambda split: -quotas[split][1])
    for split in by_remainder[: size - sum(counts.values())]:
        counts[split] += 1
    return counts


def _partition_patients(labels: dict[EyeKey, int], seed: int) -> dict[str, str]:
    diagnoses: dict[str, list[int]] = defaultdict(list)
    for (patient, _), label in labels.items():
        diagnoses[patient].append(label)
    profiles: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for patient, patient_labels in diagnoses.items():
        profiles[tuple(sorted(patient_labels))].append(patient)

    rng = random.Random(seed)
    assignments: dict[str, str] = {}
    for profile in sorted(profiles):
        patients = sorted(profiles[profile])
        rng.shuffle(patients)
        start = 0
        for split, count in _allocate_splits(len(patients)).items():
            assignments.update((patient, split) for patient in patients[start : start + count])
            start += count
    missing_splits = SPLIT_PERCENTAGES.keys() - set(assignments.values())
    if missing_splits:
        raise PapilaError(f"Pacientes insuficientes por perfil para poblar: {', '.join(sorted(missing_splits))}.")
    return assignments


def _write_manifest(output: Path, rows: list[ManifestRow]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        dir=output.parent, mode="w", encoding="utf-8", newline="", delete=False
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=list(ManifestRow.__annotations__), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temp_path.replace(output)
    finally:
        temp_path.unlink(missing_ok=True)


def prepare_papila(dataset_dir: Path, output: Path, seed: int = DEFAULT_SEED) -> PreparationResult:
    """Validate all image/label pairs and replace the output with a deterministic CSV."""
    if output.suffix.lower() != ".csv":
        raise PapilaError(f"La salida debe ser un archivo .csv: {output}")
    images = _index_images(dataset_dir / "FundusImages")
    labels, redundant = _read_labels(dataset_dir / "ClinicalData")
    for missing, description in ((images.keys() - labels.keys(), "Imágenes sin etiqueta"),
                                 (labels.keys() - images.keys(), "Etiquetas sin imagen")):
        if missing:
            examples = ", ".join(_image_id(key) for key in sorted(missing)[:5])
            raise PapilaError(f"{description} ({len(missing)}): {examples}.")

    for path in images.values():
        try:
            with Image.open(path) as image:
                if image.format != "JPEG":
                    raise PapilaError(f"La imagen {path.name} debe estar en formato JPEG.")
                image.load()
        except (OSError, Image.DecompressionBombError) as error:
            raise PapilaError(f"No se pudo decodificar la imagen {path.name}: {error}") from error

    eligible = {key: record.diagnosis for key, record in labels.items() if record.diagnosis != 2}
    if not eligible:
        raise PapilaError("No quedan imágenes elegibles tras excluir los diagnósticos sospechosos (2).")
    assignments = _partition_patients(eligible, seed)
    rows: list[ManifestRow] = [
        {
            "image_id": _image_id(key),
            "patient_id": key[0],
            "eye": key[1],
            "label": eligible[key],
            "relative_path": images[key].relative_to(dataset_dir).as_posix(),
            "split": assignments[key[0]],
        }
        for key in sorted(eligible)
    ]
    _write_manifest(output, rows)
    return PreparationResult(rows, len(images) - len(rows), redundant)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Raíz de PAPILA descomprimido.")
    parser.add_argument("--output", type=Path, required=True, help="Ruta de salida del manifiesto CSV.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Semilla de partición (por defecto 42).")
    args = parser.parse_args()
    try:
        result = prepare_papila(args.dataset_dir, args.output, args.seed)
    except (PapilaError, OSError) as error:
        parser.error(str(error))

    print(f"Semilla: {args.seed} | Imágenes originales: {len(result.rows) + result.excluded_suspects} | "
          f"Sospechosas excluidas: {result.excluded_suspects} | Elegibles: {len(result.rows)} | "
          f"Etiquetas redundantes conciliadas: {result.redundant_labels}")
    patients: set[str] = set()
    patient_total = 0
    print(f"{'Split':<8} {'Pacientes':>9} {'Imágenes':>9} {'Sano (0)':>9} {'Glaucoma (1)':>12}")
    for split in SPLIT_PERCENTAGES:
        rows = [row for row in result.rows if row["split"] == split]
        split_patients = {row["patient_id"] for row in rows}
        patient_total += len(split_patients)
        patients.update(split_patients)
        positives = sum(row["label"] for row in rows)
        print(f"{split:<8} {len(split_patients):>9} {len(rows):>9} {len(rows) - positives:>9} {positives:>12}")
    print(f"Asignaciones de pacientes repetidas entre conjuntos: {patient_total - len(patients)}")
    print(f"Manifiesto: {args.output}")


if __name__ == "__main__":
    main()
