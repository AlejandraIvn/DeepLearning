"""Exercise PAPILA preparation with synthetic JPEGs and Excel workbooks."""

import csv
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from openpyxl import Workbook
from PIL import Image

from src.prepare_papila import PapilaError, prepare_papila


class PreparePapilaTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.images = self.root / "FundusImages"
        self.clinical = self.root / "ClinicalData"
        self.output = self.root / "output" / "manifest.csv"
        self.images.mkdir()
        self.clinical.mkdir()
        self.od_rows: list[tuple[str, int]] = []
        self.os_rows: list[tuple[str, int]] = []
        self.expected: dict[str, int] = {}

        # Ocho pacientes por perfil binario. Alternar la lateralidad del caso mixto.
        mixed = [(0, 1), (1, 0)] * 4
        pairs = [(0, 0)] * 8 + mixed + [(1, 1)] * 8 + [(2, 2), (2, 1)]
        for patient, (od, os) in enumerate(pairs, start=1):
            self.od_rows.append((f"#{patient}", od))
            for eye, diagnosis in (("OD", od), ("OS", os)):
                self.os_rows.append((f"#{patient:03d} {eye.lower()}", diagnosis))
                image_id = f"RET{patient:03d}{eye}"
                self._save_image(f"{image_id}.jpg")
                if diagnosis != 2:
                    self.expected[image_id] = diagnosis
        self._write_workbook("od", self.od_rows)
        self._write_workbook("os", self.os_rows)

    def _save_image(self, name: str) -> None:
        with Image.new("RGB", (8, 8)) as image:
            image.save(self.images / name)

    def _write_workbook(self, eye: str, rows: list[tuple[str, int]]) -> None:
        workbook = Workbook()
        sheet = workbook.active
        for header in ((None, "Age", "Gender", "Diagnosis"),) * 2 + (("ID",),):
            sheet.append(header)
        for identifier, diagnosis in rows:
            sheet.append((identifier, None, None, diagnosis))
        workbook.save(self.clinical / f"patient_data_{eye}.xlsx")
        workbook.close()

    def test_one_valid_label_per_image_and_suspects_excluded(self) -> None:
        result = prepare_papila(self.root, self.output)
        self.assertEqual(result.excluded_suspects, 3)
        self.assertEqual(result.redundant_labels, 26)
        with self.output.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            self.assertEqual(reader.fieldnames, ["image_id", "patient_id", "eye", "label", "relative_path", "split"])
        self.assertEqual(len(rows), len(self.expected))
        self.assertEqual({row["image_id"]: int(row["label"]) for row in rows}, self.expected)
        self.assertTrue(all((self.root / row["relative_path"]).is_file() for row in rows))
        self.assertEqual(rows[0]["patient_id"], "001")
        self.assertEqual(rows[0]["eye"], "OD")
        self.assertNotIn("RET026OD", {row["image_id"] for row in rows})
        self.assertIn("RET026OS", {row["image_id"] for row in rows})

    def test_partition_is_grouped_and_stratified_by_profile(self) -> None:
        result = prepare_papila(self.root, self.output)
        patient_splits = defaultdict(set)
        patient_labels = defaultdict(list)
        for row in result.rows:
            patient_splits[row["patient_id"]].add(row["split"])
            patient_labels[row["patient_id"]].append(row["label"])
        self.assertEqual(len(patient_splits), 25)
        self.assertTrue(all(len(splits) == 1 for splits in patient_splits.values()))
        self.assertEqual(sorted(patient_labels["009"]), [0, 1])
        self.assertEqual(sorted(patient_labels["010"]), [0, 1])
        profiles_by_split = Counter(
            (next(iter(patient_splits[patient])), tuple(sorted(labels)))
            for patient, labels in patient_labels.items()
        )
        for split, count in (("train", 5), ("val", 1), ("test", 2)):
            for profile in ((0, 0), (0, 1), (1, 1)):
                self.assertEqual(profiles_by_split[split, profile], count)

    def test_manifest_reproducible_with_same_seed_and_reordered_labels(self) -> None:
        prepare_papila(self.root, self.output)
        first = self.output.read_bytes()
        prepare_papila(self.root, self.output, seed=42)
        self.assertEqual(self.output.read_bytes(), first)
        self._write_workbook("od", list(reversed(self.od_rows)))
        self._write_workbook("os", list(reversed(self.os_rows)))
        prepare_papila(self.root, self.output, seed=42)
        self.assertEqual(self.output.read_bytes(), first)
        prepare_papila(self.root, self.output, seed=43)
        self.assertNotEqual(self.output.read_bytes(), first)

    def test_eye_specific_workbooks_with_no_explicit_side(self) -> None:
        only_os = [(identifier.split()[0], label) for identifier, label in self.os_rows if identifier.endswith(" os")]
        self._write_workbook("os", only_os)
        result = prepare_papila(self.root, self.output)
        self.assertEqual(result.redundant_labels, 0)
        self.assertEqual({row["image_id"]: row["label"] for row in result.rows}, self.expected)

    def test_image_without_label_reports_id_and_does_not_replace_output(self) -> None:
        prepare_papila(self.root, self.output)
        previous = self.output.read_bytes()
        self._save_image("RET900OD.jpg")
        with self.assertRaisesRegex(PapilaError, r"Imágenes sin etiqueta.*RET900OD"):
            prepare_papila(self.root, self.output)
        self.assertEqual(self.output.read_bytes(), previous)

    def test_label_without_image_reports_id(self) -> None:
        (self.images / "RET025OD.jpg").unlink()
        with self.assertRaisesRegex(PapilaError, r"Etiquetas sin imagen.*RET025OD"):
            prepare_papila(self.root, self.output)

    def test_duplicate_image_reports_id(self) -> None:
        self._save_image("RET1OD.jpeg")
        with self.assertRaisesRegex(PapilaError, r"Imagen duplicada.*RET001OD"):
            prepare_papila(self.root, self.output)

    def test_duplicate_label_in_one_workbook_reports_both_rows(self) -> None:
        self._write_workbook("os", self.os_rows + [("#1 OD", 0)])
        with self.assertRaisesRegex(PapilaError, r"Etiqueta duplicada.*RET001OD.*fila 56.*fila 4"):
            prepare_papila(self.root, self.output)

    def test_conflicting_labels_report_both_sources(self) -> None:
        self.od_rows[0] = ("#1", 1)
        self._write_workbook("od", self.od_rows)
        with self.assertRaisesRegex(PapilaError, r"Etiquetas contradictorias.*RET001OD.*patient_data_od.*patient_data_os"):
            prepare_papila(self.root, self.output)

    def test_invalid_diagnosis_reports_source_and_row(self) -> None:
        self.os_rows[1] = ("#001 os", 7)
        self._write_workbook("os", self.os_rows)
        with self.assertRaisesRegex(PapilaError, r"Diagnóstico inválido.*RET001OS.*patient_data_os.*fila 5"):
            prepare_papila(self.root, self.output)

    def test_invalid_clinical_id_reports_row(self) -> None:
        self.os_rows[1] = ("#001 xx", 0)
        self._write_workbook("os", self.os_rows)
        with self.assertRaisesRegex(PapilaError, r"Identificador clínico inválido.*fila 5.*#001 xx"):
            prepare_papila(self.root, self.output)

    def test_diagnosis_with_blank_id_is_rejected_even_before_first_record(self) -> None:
        self._write_workbook("os", [("", 1)] + self.os_rows)
        with self.assertRaisesRegex(PapilaError, r"Identificador clínico inválido.*fila 4"):
            prepare_papila(self.root, self.output)

    def test_corrupt_image_reports_filename(self) -> None:
        (self.images / "RET001OD.jpg").write_bytes(b"invalid jpeg")
        with self.assertRaisesRegex(PapilaError, r"No se pudo decodificar la imagen RET001OD\.jpg"):
            prepare_papila(self.root, self.output)


if __name__ == "__main__":
    unittest.main()
