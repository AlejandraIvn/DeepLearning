"""Exercise fixed-threshold metrics and test-run safety without PAPILA images."""

import io
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src import internal_test
from src.compare import plot_curves
from src.device import select_device


class InferenceProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("count", torch.tensor(0))

    def forward(self, images):
        assert torch.is_inference_mode_enabled()
        assert not torch.is_grad_enabled()
        return images[:, 0]


class InternalEvaluationTests(unittest.TestCase):
    def test_undefined_metrics_on_single_class_or_no_positive_predictions(self) -> None:
        labels = np.array([0, 0])
        result = internal_test.binary_metrics(labels, np.array([-2., -1.]), np.array([False, False]))
        self.assertEqual(result["specificity"], 1.0)
        for name in ("recall", "precision", "f1", "f2", "roc_auc", "average_precision"):
            self.assertIsNone(result[name])
        result = internal_test.binary_metrics(np.array([0, 1]), np.array([-2., -1.]), np.array([False, False]))
        self.assertEqual(result["f2"], 0.0)
        self.assertIsNone(result["precision"])
        self.assertEqual(result["roc_auc"], 1.0)

    def test_patient_bootstrap_keeps_paired_eyes_and_is_reproducible(self) -> None:
        patients = ["a", "a", "b", "b"]
        labels = np.array([0, 0, 1, 1])
        logits = np.array([-2., -1., 1., 2.])
        point = internal_test.binary_metrics(labels, logits, logits >= 0)
        estimates = {"CNN": point, "ResNet18": point,
                     "ResNet18_minus_CNN": dict.fromkeys(internal_test.METRICS, 0)}
        arguments = (patients, labels, {"CNN": logits, "ResNet18": logits},
                     {"CNN": logits >= 0, "ResNet18": logits >= 0}, estimates)
        kwargs = {"replicates": 40, "seed": 42, "confidence": 0.95}
        result = internal_test.bootstrap_intervals(*arguments, **kwargs)
        self.assertEqual(result, internal_test.bootstrap_intervals(*arguments, **kwargs))
        draws = np.random.default_rng(42).integers(2, size=(40, 2))
        single_class = int((draws[:, 0] == draws[:, 1]).sum())
        self.assertEqual(result["CNN"]["roc_auc"]["invalid_resamples"], single_class)
        self.assertEqual(result["CNN"]["accuracy"]["invalid_resamples"], 0)
        self.assertEqual(result["ResNet18_minus_CNN"]["recall"]["lower"], 0.0)
        self.assertEqual(result["ResNet18_minus_CNN"]["recall"]["upper"], 0.0)

    def test_forward_requires_eval_and_score_detects_buffer_updates(self) -> None:
        model = InferenceProbe()
        inputs = torch.ones((2, 1))
        with self.assertRaisesRegex(ValueError, "modo evaluación"):
            internal_test._forward(model, inputs)
        model.eval()
        loader = DataLoader(TensorDataset(inputs, torch.zeros(2)), batch_size=1)
        device = select_device(prefer_cuda=False)
        _, logits = internal_test._score(model, loader, device)
        np.testing.assert_array_equal(logits, [1., 1.])

        class MutatingProbe(InferenceProbe):
            def forward(self, images):
                self.count.add_(1)
                return super().forward(images)

        with self.assertRaisesRegex(ValueError, "buffers cambiaron"):
            internal_test._score(MutatingProbe().eval(), loader, device)

    def test_cli_requires_explicit_confirmation_or_preflight(self) -> None:
        args = ["internal_test", "--dataset-dir", "unused", "--manifest", "unused"]
        with patch("sys.argv", args), patch.object(internal_test, "run_test") as run, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                internal_test.main()
        self.assertEqual(raised.exception.code, 2)
        run.assert_not_called()

    def test_existing_output_blocks_another_test_pass_before_loading_inputs(self) -> None:
        with TemporaryDirectory() as temporary, patch.object(internal_test, "OUTPUT", Path(temporary)), \
                patch.object(internal_test, "_prepare") as prepare:
            with self.assertRaises(FileExistsError):
                internal_test.run_test({}, Path("unused"), Path("unused"))
        prepare.assert_not_called()

    def test_curves_accept_integer_labels_from_saved_predictions(self) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "curves.png"
            plot_curves(output, torch.tensor([0, 1]), {"Modelo": torch.tensor([-1., 1.])},
                        {"Modelo": (1.0, 1.0)}, split_label="test interno")
            self.assertGreater(output.stat().st_size, 0)
