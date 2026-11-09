"""Grad-CAM geometry, rendering, validation selection and frozen FP32 autograd."""

import io
import unittest
from pathlib import PurePosixPath
from unittest.mock import patch

import torch
from PIL import Image, ImageDraw

from src.data import ImageRecord, Letterbox
from src.explain import align_heatmap, explain_image, letterbox_content_box, normalize_heatmap, summarize_regions
from src.explain_validation import select_examples
from src.explain_visualization import overlay_png
from src.predict import checkpoint_transform
from src.pretrained import PapilaResNet18


CHECKPOINT = {
    "architecture": {"image_size": 224},
    "preprocessing": {"geometry": "letterbox"},
    "normalization": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
}


class ExplainTests(unittest.TestCase):
    def test_content_box_matches_pillow_padding_even_with_odd_rounding(self) -> None:
        for size, expected in (
            ((224, 145), (0, 40, 224, 185)), ((145, 224), (40, 0, 185, 224)),
            ((224, 147), (0, 38, 224, 185)), ((50, 50), (0, 0, 224, 224)),
        ):
            with self.subTest(size=size):
                self.assertEqual(letterbox_content_box(size, 224), expected)
                mask = Letterbox(224)(Image.new("L", size, 255))
                self.assertEqual(mask.getbbox(), expected)

    def test_crop_excludes_padding_and_preserves_native_orientation(self) -> None:
        for original in ((160, 80), (80, 160)):
            with self.subTest(original=original):
                left, top, right, bottom = box = letterbox_content_box(original, 8)
                padded = torch.full((8, 8), 50.0)
                padded[top:bottom, left:right] = 0
                padded[top:top + (bottom - top) // 2, left + (right - left) // 2:right] = 1
                aligned = align_heatmap(padded, 8, box, original)
                width, height = original
                self.assertEqual(aligned.shape, (height, width))
                self.assertEqual(aligned[height // 4, 3 * width // 4].item(), 1)
                self.assertEqual(aligned[height // 4, width // 4].item(), 0)
                self.assertEqual(aligned[3 * height // 4, 3 * width // 4].item(), 0)

    def test_normalization_handles_zero_uniform_and_nonfinite_maps(self) -> None:
        for raw in ([0., 0.], [2., 2.], [2., float("nan")], [float("inf"), float("nan")]):
            with self.subTest(raw=raw):
                torch.testing.assert_close(normalize_heatmap(torch.tensor(raw)), torch.zeros(2))
        mixed = torch.tensor([float("inf"), 1., 2., 3., float("nan"), -float("inf")])
        torch.testing.assert_close(normalize_heatmap(mixed), torch.tensor([0., 0., 0.5, 1., 0., 0.]))
        box = letterbox_content_box((161, 87), 224)
        aligned = align_heatmap(torch.full((7, 7), 0.1), 224, box, (161, 87))
        self.assertEqual(aligned.count_nonzero().item(), 0)

    def test_overlay_png_is_native_size_and_transparent_where_no_signal(self) -> None:
        original = Image.new("RGB", (3, 2), (12, 34, 56))
        heatmap = torch.zeros((2, 3))
        heatmap[0, 2] = 1
        with Image.open(io.BytesIO(overlay_png(original, heatmap))) as png:
            self.assertEqual(png.size, original.size)
            self.assertEqual(png.getpixel((0, 1)), original.getpixel((0, 1)))
            self.assertNotEqual(png.getpixel((2, 0)), original.getpixel((2, 0)))
        with Image.open(io.BytesIO(overlay_png(original, heatmap, alpha=0))) as png:
            self.assertEqual(png.tobytes(), original.tobytes())

    def test_region_summary_accounts_for_area_and_no_signal(self) -> None:
        mask = torch.tensor([[False, True], [False, False]])
        result = summarize_regions(torch.tensor([[0., 1.], [0., 0.]]), {"roi": mask})
        self.assertEqual(result["roi"], {"area_fraction": 0.25, "cam_fraction": 1., "cam_to_area_ratio": 4.})
        self.assertIsNone(summarize_regions(torch.zeros(2, 2), {"roi": mask})["roi"]["cam_fraction"])
        with self.assertRaisesRegex(ValueError, "Máscara booleana no alineada"):
            summarize_regions(torch.zeros(2, 2), {"roi": mask[:1]})

    def test_selection_uses_first_id_and_leaves_empty_categories_absent(self) -> None:
        def record(image_id, label):
            return ImageRecord(image_id, image_id[:3], "OD", label, PurePosixPath(f"{image_id}.jpg"))

        rows = [(record("003OD", 1), 0.99), (record("001OD", 1), 0.35),
                (record("002OD", 0), 0.1), (record("004OD", 1), 0.1)]
        selected = select_examples(rows, 0.35)
        self.assertEqual(selected, select_examples(reversed(rows), 0.35))
        self.assertEqual({key: row.image_id for key, (row, _) in selected.items()},
                         {"TP": "001OD", "TN": "002OD", "FN": "004OD"})


class ResNetGradCAMTests(unittest.TestCase):
    def setUp(self) -> None:
        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(21)
            self.model = PapilaResNet18(pretrained=False).eval().requires_grad_(False)
        self.image = Image.new("RGB", (161, 87), (40, 100, 150))
        ImageDraw.Draw(self.image).rectangle((90, 10, 120, 60), fill=(210, 0, 0))

    def _verify_cam(self, device: str, amp_dtype: torch.dtype) -> None:
        model = self.model.to(device)
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        layer = model.backbone.layer4[-1]
        hooks = tuple(layer._forward_hooks)
        pixels = checkpoint_transform(CHECKPOINT)(self.image).unsqueeze(0).to(device)
        with torch.inference_mode():
            baseline = model(pixels).sigmoid().item()
        with torch.no_grad(), torch.autocast(device, dtype=amp_dtype):
            result = explain_image(model, self.image, CHECKPOINT, threshold=0.35)
        self.assertEqual(result.feature_shape, (7, 7))
        self.assertEqual(result.heatmap.shape, (87, 161))
        self.assertEqual(result.heatmap.dtype, torch.float32)
        self.assertEqual(result.heatmap.device.type, "cpu")
        self.assertFalse(result.heatmap.requires_grad)
        self.assertTrue(result.heatmap.isfinite().all().item())
        self.assertGreater(result.heatmap.max().item(), 0)
        self.assertLessEqual(result.heatmap.max().item(), 1)
        self.assertGreaterEqual(result.heatmap.min().item(), 0)
        self.assertAlmostEqual(result.score_glaucoma, baseline, places=6)
        self.assertEqual(result.label, int(baseline >= 0.35))
        self.assertEqual(tuple(layer._forward_hooks), hooks)
        self.assertTrue(all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()))
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, before[name]), name)

    def test_frozen_resnet18_on_cpu(self) -> None:
        self._verify_cam("cpu", torch.bfloat16)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA no disponible")
    def test_frozen_resnet18_on_cuda(self) -> None:
        self._verify_cam("cuda", torch.float16)

    def test_negative_prediction_still_targets_positive_logit(self) -> None:
        self.model.fc_out.weight.fill_(1 / 512)
        self.model.fc_out.bias.fill_(-10)
        result = explain_image(self.model, self.image, CHECKPOINT, threshold=0.35)
        self.assertEqual(result.label, 0)
        self.assertGreater(result.heatmap.max().item(), 0)

    def test_zero_signal_produces_zero_map(self) -> None:
        self.model.fc_out.weight.zero_()
        result = explain_image(self.model, self.image, CHECKPOINT, threshold=0.35)
        self.assertEqual(result.heatmap.count_nonzero().item(), 0)

    def test_hook_is_removed_if_forward_fails(self) -> None:
        layer = self.model.backbone.layer4[-1]
        with layer.register_forward_hook(lambda *_: None):
            hooks = tuple(layer._forward_hooks)
            with patch.object(self.model.fc_out, "forward", side_effect=RuntimeError("error de prueba")):
                with self.assertRaisesRegex(RuntimeError, "error de prueba"):
                    explain_image(self.model, self.image, CHECKPOINT, threshold=0.35)
            self.assertEqual(tuple(layer._forward_hooks), hooks)
