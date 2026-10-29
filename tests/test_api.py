"""API tests against a temporary runs directory.

The model used here is the randomly initialised test fixture; tests only check plumbing,
schemas and error handling, never prediction quality.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.schemas import PredictionResponse
from api.settings import Settings
from neurolens.inference.image_io import InvalidImageError, decode_image
from tests.conftest import make_image, png_bytes


def _client(runs_dir: Path, **kwargs) -> TestClient:
    settings = Settings(runs_dir=runs_dir, reports_dir=runs_dir / "reports", device="cpu", **kwargs)
    return TestClient(create_app(settings))


@pytest.fixture
def client(tiny_checkpoint: Path):
    with _client(tiny_checkpoint.parent) as c:
        yield c


@pytest.fixture
def empty_client(tmp_path: Path):
    with _client(tmp_path / "no_runs") as c:
        yield c


def _upload(data: bytes, name: str = "scan.png", ctype: str = "image/png") -> dict:
    return {"file": (name, data, ctype)}


def test_health_and_models_ready(client: TestClient) -> None:
    health = client.get("/health").json()
    assert health["ready"] is True and health["models_loaded"] == ["test_run"]
    models = client.get("/models").json()
    assert {a["name"] for a in models["architectures"]} == {
        "custom_cnn",
        "resnet18",
        "efficientnet_b0",
    }
    assert models["models"][0]["name"] == "test_run" and models["models"][0]["loaded"]
    assert "path" not in models["models"][0]  # no filesystem paths exposed


def test_predict_response_schema(client: TestClient) -> None:
    response = client.post("/predict", files=_upload(png_bytes(make_image(1, 80))))
    assert response.status_code == 200, response.text
    body = PredictionResponse(**response.json())
    assert abs(sum(body.raw_probabilities.values()) - 1.0) < 1e-5
    assert body.calibrated_probabilities is None  # no calibration.json for the fixture
    assert body.predicted_class in body.class_names
    assert "not a medical diagnosis" in body.disclaimer
    assert body.model.name == "test_run"
    assert body.preprocessing["image_size"] == 64
    assert any("calibration" in w for w in body.warnings)


def test_predict_uses_calibration_when_present(client: TestClient, tiny_checkpoint: Path) -> None:
    from neurolens.utils import write_json

    write_json(
        tiny_checkpoint / "calibration.json", {"temperature": 2.0, "model_version": "test_run-e001"}
    )
    with _client(tiny_checkpoint.parent) as fresh:
        body = fresh.post("/predict", files=_upload(png_bytes(make_image(2, 80)))).json()
    assert body["model"]["temperature"] == 2.0
    assert body["calibrated_probabilities"] is not None
    assert body["uncertainty"]["probabilities_used"] == "calibrated"


def test_explain_returns_images(client: TestClient) -> None:
    response = client.post(
        "/explain", files=_upload(png_bytes(make_image(3, 80), "JPEG"), "a.jpg", "image/jpeg")
    )
    assert response.status_code == 200, response.text
    explanation = response.json()["explanation"]
    for key in (
        "original_png_base64",
        "heatmap_png_base64",
        "overlay_png_base64",
        "legend_png_base64",
    ):
        assert base64.b64decode(explanation[key]).startswith(b"\x89PNG")
    assert "does not" in explanation["interpretation_note"]


def test_invalid_uploads(client: TestClient) -> None:
    assert client.post("/predict", files=_upload(b"")).status_code == 400
    assert (
        client.post("/predict", files=_upload(b"hello world", "x.txt", "text/plain")).status_code
        == 400
    )
    gif = png_bytes(make_image(4, 64).convert("P"), "GIF")
    assert client.post("/predict", files=_upload(gif, "x.gif", "image/gif")).status_code == 415
    tiny = png_bytes(make_image(5, 8))
    assert client.post("/predict", files=_upload(tiny)).status_code == 400
    assert (
        client.post(
            "/predict", files=_upload(png_bytes(make_image(6, 64))), params={"model": "../../etc"}
        ).status_code
        == 404
    )
    assert client.post("/predict").status_code == 422


def test_upload_size_limit(tiny_checkpoint: Path) -> None:
    with _client(tiny_checkpoint.parent, max_upload_mb=0.001) as small:
        response = small.post("/predict", files=_upload(png_bytes(make_image(7, 128))))
    assert response.status_code == 413


def test_missing_checkpoint_reports_not_ready(empty_client: TestClient) -> None:
    health = empty_client.get("/health").json()
    assert health["status"] == "ok" and health["ready"] is False
    response = empty_client.post("/predict", files=_upload(png_bytes(make_image(8, 64))))
    assert response.status_code == 503
    assert "No trained model" in response.json()["detail"]
    assert empty_client.get("/models").json()["models"] == []


def test_metrics_separates_offline_and_live(client: TestClient) -> None:
    client.post("/predict", files=_upload(png_bytes(make_image(9, 64))))
    body = client.get("/metrics").json()
    assert body["offline_evaluation"] == [{"run": "test_run", "status": "not evaluated"}]
    assert body["live_service"]["requests"]["predict"] >= 1
    assert body["model_comparison"] is None


def test_decode_image_rejects_garbage() -> None:
    with pytest.raises(InvalidImageError):
        decode_image(b"\x89PNG\r\n\x1a\n broken")
    assert decode_image(png_bytes(make_image(10, 64))).mode == "RGB"
