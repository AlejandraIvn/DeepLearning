"""Exercise Render startup, private checkpoint retrieval and public uploads."""

import io
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import URLError

from fastapi.testclient import TestClient

from backend import main


REMOTE = {"RENDER": "true"}
CHECKPOINT_URL = "https://example.invalid/best.pt?signature=private"


class DeploymentTests(unittest.TestCase):
    def test_render_requires_the_checkpoint_secret_file(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, REMOTE):
            with patch.object(main, "CHECKPOINT_URL_FILE", Path(directory) / "missing"), self.assertLogs(main.logger) as logs:
                with self.assertRaisesRegex(RuntimeError, "demostración requiere"), TestClient(main.create_app()):
                    self.fail("El servicio arrancó sin el checkpoint.")
            self.assertIn("glaucoai-checkpoint-url", " ".join(logs.output))

    def test_public_analyze_accepts_cross_origin_uploads(self):
        process = Mock(return_value=main.Analysis(
            score_glaucoma=0.42, label=1, classification="glaucoma", threshold=0.35,
            heatmap_png="heatmap", overlay_png="overlay", target_layer="layer", has_signal=True,
        ))
        service = SimpleNamespace(device="cpu", threshold=0.35, process=process)
        headers = {"Origin": "https://demo.example", "Content-Type": "image/jpeg"}
        with patch.dict(os.environ, REMOTE), patch.object(main, "load_model", return_value=service):
            with TestClient(main.create_app()) as client:
                self.assertEqual(client.get("/api/health").status_code, 200)
                preflight = client.options("/api/analyze", headers={
                    "Origin": headers["Origin"], "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                })
                self.assertEqual(preflight.status_code, 200)
                self.assertEqual(preflight.headers["access-control-allow-origin"], "*")
                response = client.post("/api/analyze", headers=headers, content=b"bytes")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["access-control-allow-origin"], "*")
                self.assertEqual(response.headers["cache-control"], "no-store")
                process.assert_called_once_with(b"bytes", "JPEG", explain=True)

    def test_download_is_bounded_and_redacts_signed_url_on_failure(self):
        opener = Mock()
        with TemporaryDirectory() as directory:
            secret = Path(directory) / "glaucoai-checkpoint-url"
            secret.write_text(CHECKPOINT_URL + "\n", encoding="utf-8")
            with patch.object(main, "CHECKPOINT_URL_FILE", secret), \
                    patch.object(main, "build_opener", return_value=opener), patch.object(main, "MAX_CHECKPOINT_BYTES", 4):
                opener.open.return_value = io.BytesIO(b"abcd")
                self.assertEqual(main._remote_checkpoint(), b"abcd")
                opener.open.assert_called_once_with(CHECKPOINT_URL, timeout=30)
                opener.open.return_value = io.BytesIO(b"abcde")
                with self.assertRaisesRegex(ValueError, "64 MiB"):
                    main._remote_checkpoint()
                opener.open.side_effect = URLError(CHECKPOINT_URL)
                with self.assertRaises(ValueError) as error:
                    main._remote_checkpoint()
                self.assertNotIn("signature", str(error.exception))

    def test_remote_model_hash_is_checked_before_deserializing(self):
        with patch.dict(os.environ, REMOTE), patch.object(main, "_remote_checkpoint", return_value=b"untrusted"):
            with patch.object(main, "load_checkpoint") as deserialize:
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    main.load_model()
                deserialize.assert_not_called()
