"""Check the historical CNN, selection and local inference without real PAPILA."""

import csv
import hashlib
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from src import train
from src.data import Normalization, make_image_transform
from src.device import select_device
from src.model import PapilaCNN, load_checkpoint
from src.predict import predict_image
from src.train import EarlyStopping, TrainConfig, run_training, validation_metrics


class ModelTests(unittest.TestCase):
    def test_historical_shape_and_unbounded_logits(self) -> None:
        model = PapilaCNN()
        self.assertEqual(model.fc1.weight.shape, (128, 64 * 24 * 24))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 4_738_241)
        with torch.no_grad():
            model.fc_out.weight.zero_()
            model.fc_out.bias.fill_(-2)
        logits = model(torch.zeros(1, 3, 96, 96))
        self.assertEqual(logits.shape, (1,))
        torch.testing.assert_close(logits, torch.tensor([-2.0]))

    def test_cpu_forward_and_backward(self) -> None:
        model = PapilaCNN(image_size=8)
        logits = model(torch.ones(2, 3, 8, 8))
        torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros(2)).backward()
        self.assertTrue(model.conv1.weight.grad.isfinite().all().item())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA no disponible")
    def test_cuda_amp_forward_and_backward(self) -> None:
        device = select_device()
        model = PapilaCNN(image_size=8).to(device.device)
        optimizer = torch.optim.Adam(model.parameters())
        scaler = device.make_grad_scaler()
        images, labels = device.move_batch((torch.zeros(2, 3, 8, 8), torch.tensor([0.0, 1.0])))
        with device.autocast():
            logits = model(images)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        self.assertEqual(logits.shape, labels.shape)
        self.assertEqual(logits.dtype, torch.float16)
        self.assertEqual(loss.dtype, torch.float32)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        self.assertTrue(model.fc_out.weight.grad.isfinite().all().item())


class TrainingTests(unittest.TestCase):
    def test_early_stopping_resets_on_strict_improvement(self) -> None:
        stopping = EarlyStopping(patience=2)
        self.assertEqual(stopping.step(0.5), (True, False))
        self.assertEqual(stopping.step(0.6), (False, False))
        self.assertEqual(stopping.step(0.4), (True, False))
        self.assertEqual(stopping.step(0.4), (False, False))
        self.assertEqual(stopping.step(0.5), (False, True))
        self.assertEqual(stopping.best_loss, 0.4)

    def test_metrics_with_glaucoma_positive_and_single_class(self) -> None:
        metrics = validation_metrics(torch.tensor([0, 0, 1, 1]), torch.tensor([-2., 1., -1., 2.]), 0.5)
        self.assertEqual([metrics[f"val_{name}"] for name in ("tn", "fp", "fn", "tp")], [1, 1, 1, 1])
        self.assertEqual(metrics["val_f1"], 0.5)
        self.assertEqual(metrics["val_roc_auc"], 0.75)
        self.assertAlmostEqual(metrics["val_pr_auc"], 19 / 24)
        single = validation_metrics(torch.tensor([0, 0]), torch.tensor([-2., 1.]), 0.5)
        self.assertIsNone(single["val_roc_auc"])
        self.assertIsNone(single["val_pr_auc"])

    def test_one_epoch_updates_weights_and_reloads_same_predictions_without_test_images(self) -> None:
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        self.addCleanup(torch.use_deterministic_algorithms, torch.are_deterministic_algorithms_enabled())
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("image_id", "patient_id", "eye", "label", "relative_path", "split"))
                for patient, split in enumerate(("train", "train", "val", "test"), start=1):
                    for eye, label in (("OD", 0), ("OS", 1)):
                        image_id = f"RET{patient:03d}{eye}"
                        path = f"{image_id}.png"
                        writer.writerow((image_id, f"{patient:03d}", eye, label, path, split))
                        if split != "test":
                            with Image.new("RGB", (8, 8), color=(16 + 150 * label, 40, 100)) as image:
                                image.save(root / path)
            run_dir = root / "run"
            config = TrainConfig(image_size=8, batch_size=2, epochs=1)
            probe = torch.full((1, 3, 8, 8), 0.5)
            before_save = []
            validate = train._validate

            def capture(model, *args):
                metrics = validate(model, *args)
                with torch.inference_mode():
                    before_save.append(torch.sigmoid(model(probe)))
                return metrics

            with patch.object(train, "_validate", side_effect=capture), redirect_stdout(io.StringIO()):
                summary = run_training(root, manifest, run_dir, config, select_device(prefer_cuda=False))
            self.assertEqual(summary["epochs_run"], 1)
            self.assertEqual(summary["best_epoch"], 1)
            self.assertTrue(summary["updated_head"])
            self.assertEqual((summary["train_images"], summary["val_images"]), (4, 2))
            self.assertTrue((run_dir / "loss.png").is_file())
            with (run_dir / "history.csv").open(encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)
            model, payload = load_checkpoint(run_dir / "best.pt")
            self.assertEqual(payload["manifest_sha256"], hashlib.sha256(manifest.read_bytes()).hexdigest())
            self.assertEqual(payload["config"]["seed"], 42)
            self.assertEqual(tuple(payload["classes"]), ("sano", "glaucoma"))
            self.assertFalse(model.training)
            with torch.inference_mode():
                after_reload = torch.sigmoid(model(probe))
            torch.testing.assert_close(before_save[0], after_reload, rtol=0, atol=0)

            path = root / "RET003OS.png"
            transform = make_image_transform(config.image_size, Normalization(**payload["normalization"]))
            with Image.open(path) as image, torch.inference_mode():
                expected = torch.sigmoid(model(transform(image).unsqueeze(0))).item()
            result = predict_image(run_dir / "best.pt", path)
            self.assertEqual(result["score_glaucoma"], expected)
            self.assertEqual(result["label"], int(expected >= 0.5))
            with self.assertRaisesRegex(FileExistsError, "ya existe"):
                run_training(root, manifest, run_dir, config)
