"""Exercise remote startup, private checkpoint retrieval and the upload gate."""

import io
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import URLError

from fastapi.testclient import TestClient

from backend import main


TOKEN = "demo-secret-for-testing-only-123456789"
REMOTE = {"GLAUCOAI_REMOTE": "1", "GLAUCOAI_ACCESS_TOKEN": TOKEN}
CHECKPOINT_URL = "https://example.invalid/best.pt?signature=private"


class DeploymentTests(unittest.TestCase):
    def test_render_requires_a_token_before_loading_weights(self):
        with patch.dict(os.environ, {"RENDER": "true", "GLAUCOAI_ACCESS_TOKEN": ""}):
            with patch.object(main, "load_model") as load:
                with self.assertRaisesRegex(RuntimeError, "GLAUCOAI_ACCESS_TOKEN"), TestClient(main.create_app()):
                    self.fail("El servicio arrancó sin autenticación.")
                load.assert_not_called()

    def test_uploads_require_a_token_on_prefixed_and_legacy_routes(self):
        process = Mock(return_value=main.Prediction(score_glaucoma=0.42, label=1,
                                                   classification="glaucoma", threshold=0.35))
        service = SimpleNamespace(device="cpu", threshold=0.35, process=process)
        with patch.dict(os.environ, REMOTE), patch.object(main, "load_model", return_value=service):
            with TestClient(main.create_app()) as client:
                self.assertEqual(client.get("/api/health").status_code, 200)
                for prefix in ("", "/api"):
                    for route in ("/predict", "/explain", "/analyze"):
                        denied = client.post(prefix + route, headers={"Authorization": "Bearer wrong"})
                        self.assertEqual(denied.status_code, 401)
                process.assert_not_called()
                response = client.post("/api/predict", headers={"Authorization": f"Bearer {TOKEN}",
                                                                 "Content-Type": "image/jpeg"}, content=b"bytes")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["cache-control"], "no-store")
                process.assert_called_once_with(b"bytes", "JPEG", explain=False)

    def test_download_is_bounded_and_redacts_signed_url_on_failure(self):
        opener = Mock()
        with patch.dict(os.environ, {"GLAUCOAI_CHECKPOINT_URL": CHECKPOINT_URL}):
            with patch.object(main, "build_opener", return_value=opener), patch.object(main, "MAX_CHECKPOINT_BYTES", 4):
                opener.open.return_value = io.BytesIO(b"abcd")
                self.assertEqual(main._remote_checkpoint(), b"abcd")
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
