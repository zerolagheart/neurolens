"""The PyTorch-free Vercel demo (deploy/vercel) must reproduce the PyTorch predictor."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from fastapi.testclient import TestClient

from neurolens.inference.predictor import Predictor
from neurolens.models.factory import ARCHITECTURES, build_model
from neurolens.optimization.web_export import CamExportModule, export_web_bundle, split_for_cam
from tests.conftest import make_image, png_bytes

WEB_MODULE = Path(__file__).resolve().parents[1] / "deploy" / "vercel" / "api" / "index.py"


def load_web_module(model_dir: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.setenv("NEUROLENS_WEB_MODEL_DIR", str(model_dir))
    spec = importlib.util.spec_from_file_location("neurolens_web_test", WEB_MODULE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "neurolens_web_test", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bundle(tiny_checkpoint: Path, tmp_path: Path) -> Path:
    out = tmp_path / "web_model"
    export_web_bundle(tiny_checkpoint, out, n_parity=2)
    return out


@pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
def test_split_reproduces_logits(arch: str) -> None:
    model = build_model(arch, 4).eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.inference_mode():
        logits, features = CamExportModule(*split_for_cam(model, arch)).eval()(x)
        assert torch.allclose(logits, model(x), atol=1e-5)
    assert features.ndim == 4


def test_bundle_contents(bundle: Path) -> None:
    import json

    meta = json.loads((bundle / "meta.json").read_text())
    for name in ("model.onnx", "cam_weights.npy", "colormap.npy"):
        assert (bundle / name).exists()
    assert meta["class_names"] == ["glioma", "meningioma", "notumor", "pituitary"]
    assert meta["parity"]["max_abs_cam_diff_vs_gradcam"] <= meta["parity"]["cam_tolerance"]
    assert np.load(bundle / "colormap.npy").shape == (256, 3)


def test_resize_matches_torch_interpolate(bundle: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    web = load_web_module(bundle, monkeypatch)
    cam = np.random.default_rng(0).random((7, 7)).astype(np.float32)
    expected = F.interpolate(torch.from_numpy(cam)[None, None], size=(64, 64), mode="bilinear",
                             align_corners=False)[0, 0].numpy()  # fmt: skip
    assert np.allclose(web.resize_bilinear(cam, 64), expected, atol=1e-6)


def test_web_prediction_matches_pytorch(
    bundle: Path, tiny_checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    web = load_web_module(bundle, monkeypatch)
    image = make_image(11, size=96)
    out = web.analyze_bytes(png_bytes(image), web.get_engine())
    ref, _ = Predictor.from_run_dir(tiny_checkpoint, device="cpu").explain(image.convert("RGB"))
    web_probs = [out["prediction"]["raw_probabilities"][c] for c in ref.class_names]
    assert np.allclose(web_probs, ref.raw_probabilities, atol=1e-4)
    assert out["prediction"]["predicted_class"] == ref.predicted_class
    assert "not a medical diagnosis" in out["prediction"]["disclaimer"]


def test_web_api_endpoints_and_errors(bundle: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    web = load_web_module(bundle, monkeypatch)
    client = TestClient(web.app)
    assert client.get("/api/health").json()["ready"] is True
    assert client.get("/api/info").json()["model_name"] == "test_run"
    ok = client.post(
        "/api/analyze", files={"file": ("a.png", png_bytes(make_image(3, 80)), "image/png")}
    )
    assert ok.status_code == 200
    assert ok.json()["explanation"]["overlay_png_base64"]
    assert (
        client.post("/api/analyze", files={"file": ("x.txt", b"hello", "text/plain")}).status_code
        == 400
    )
    gif = png_bytes(make_image(4, 64).convert("P"), "GIF")
    assert (
        client.post("/api/analyze", files={"file": ("x.gif", gif, "image/gif")}).status_code == 415
    )
    monkeypatch.setattr(web, "MAX_UPLOAD_BYTES", 100)
    big = png_bytes(make_image(5, 128))
    assert (
        client.post("/api/analyze", files={"file": ("b.png", big, "image/png")}).status_code == 413
    )


def test_missing_bundle_returns_503(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    web = load_web_module(tmp_path / "nothing", monkeypatch)
    client = TestClient(web.app)
    assert client.get("/api/health").json()["ready"] is False
    response = client.post(
        "/api/analyze", files={"file": ("a.png", png_bytes(make_image(6, 64)), "image/png")}
    )
    assert response.status_code == 503
