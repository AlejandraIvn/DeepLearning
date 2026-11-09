"""Exercise compiled-mode routing and the frozen checkpoint on supported devices."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient
from PIL import Image

from backend import main
from test_backend import upload


SPEC = json.loads(main.CONFIG.read_text(encoding="utf-8"))["models"]["ResNet18"]
CHECKPOINT = main.CHECKPOINT
ORIGIN = "http://127.0.0.1:8000"


def fixture_dist(root):
    dist = root / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text('<div id="root"></div><script src="/assets/app.js"></script>', encoding="utf-8")
    (dist / "assets" / "app.js").write_text('fetch("/api/health")', encoding="utf-8")
    return dist


class DemoStartupTests(unittest.TestCase):
    def test_missing_frontend_is_reported_before_server_start(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "npm.cmd run build"):
                main.create_app(Path(directory))

    def test_demo_refuses_to_start_with_missing_or_mismatched_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = main.create_app(fixture_dist(root))
            path = root / "best.pt"
            with patch.object(main, "CHECKPOINT", path), patch.object(main, "load_checkpoint") as deserialize:
                for checkpoint_present, message in ((False, "Falta el checkpoint"), (True, "SHA-256")):
                    if checkpoint_present:
                        path.write_bytes(b"different model")
                    with self.subTest(checkpoint_present=checkpoint_present), self.assertLogs(main.logger) as logs:
                        with self.assertRaisesRegex(RuntimeError, "demostración requiere"):
                            with TestClient(app, base_url=ORIGIN):
                                self.fail("La API inició sin el modelo congelado.")
                        self.assertIn(message, " ".join(logs.output))
                deserialize.assert_not_called()


class DemoCheckpointTests(unittest.TestCase):
    def _verify_demo(self, *, cpu: bool):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"GLAUCOAI_CPU": str(int(cpu))}):
            app = main.create_app(fixture_dist(Path(directory)))
            with TestClient(app, base_url=ORIGIN) as client:
                page = client.get("/")
                self.assertEqual(page.status_code, 200)
                self.assertIn('/assets/app.js', page.text)
                self.assertIn("default-src 'self'", page.headers["content-security-policy"])
                self.assertEqual(client.get("/assets/app.js").text, 'fetch("/api/health")')
                self.assertEqual(client.get("/assets/missing.js").status_code, 404)
                self.assertEqual(client.get("/", headers={"Host": "example.com"}).status_code, 400)
                health = client.get("/api/health")
                self.assertEqual(health.status_code, 200)
                self.assertEqual(health.headers["cache-control"], "no-store")
                self.assertEqual(health.json()["device"], "cpu" if cpu else "cuda:0")
                self.assertEqual(health.json()["threshold"], 0.35)
                self.assertEqual(health.json(), client.get("/health").json())
                self.assertEqual(client.get("/api/missing").status_code, 404)
                service = app.state.service
                self.assertNotIn("state_dict", service.checkpoint)
                before = {name: tensor.cpu().clone() for name, tensor in service.model.state_dict().items()}
                headers = {"Content-Type": "image/png"}
                for image in (upload(100, 150), upload(190, 120), upload(100, 150)):
                    prediction = client.post("/api/predict", headers=headers, content=image)
                    analyzed = client.post("/api/analyze", headers=headers, content=image)
                    self.assertEqual((prediction.status_code, analyzed.status_code), (200, 200))
                    self.assertAlmostEqual(prediction.json()["score_glaucoma"], analyzed.json()["score_glaucoma"], places=6)
                    self.assertEqual(prediction.json()["label"], analyzed.json()["label"])
                    self.assertTrue(analyzed.json()["overlay_png"].startswith("iVBORw0KGgo"))

                broken = client.post("/api/analyze", headers=headers, content=b"corrupt")
                self.assertEqual(broken.status_code, 422)
                with patch.object(main, "_decode_image", return_value=Image.new("RGB", (48, 48))) as decode:
                    with patch.object(main, "explain_image", side_effect=RuntimeError("failed")), self.assertLogs(main.logger):
                        self.assertEqual(client.post("/api/analyze", headers=headers, content=upload()).status_code, 500)
                    with self.assertRaises(ValueError):
                        decode.return_value.getpixel((0, 0))
                self.assertEqual(client.post("/api/analyze", headers=headers, content=upload()).status_code, 200)
                self.assertFalse(service.model.backbone.layer4[-1]._forward_hooks)
                for name, tensor in service.model.state_dict().items():
                    self.assertTrue(torch.equal(before[name], tensor.cpu()), name)
            self.assertIsNone(app.state.service)
        with CHECKPOINT.open("rb") as stream:
            self.assertEqual(hashlib.file_digest(stream, "sha256").hexdigest(), SPEC["sha256"])

    def test_compiled_api_with_frozen_model_on_cpu(self):
        self._verify_demo(cpu=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA no disponible")
    def test_compiled_api_with_frozen_model_on_cuda(self):
        self._verify_demo(cpu=False)
