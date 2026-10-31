"""NeuroLens web demo: a PyTorch-free serverless function for Vercel.

Runs the exported ONNX model (``model/`` produced by ``neurolens export-web``) with ONNX
Runtime, and computes Grad-CAM from the exported feature maps and final linear weights. It
mirrors the main project's preprocessing (RGB, bilinear resize, ImageNet normalisation),
temperature scaling, entropy and abstention policy.

Research and educational use only; not a medical device.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

MODEL_DIR = Path(
    os.environ.get("NEUROLENS_WEB_MODEL_DIR", Path(__file__).resolve().parents[1] / "model")
)
# Vercel rejects request bodies above 4.5 MB, so the default stays below that.
MAX_UPLOAD_BYTES = int(float(os.environ.get("NEUROLENS_MAX_UPLOAD_MB", "4")) * 1024 * 1024)
SUPPORTED_FORMATS = {"JPEG", "PNG", "BMP", "TIFF"}
MAX_PIXELS = 40_000_000
MIN_SIDE = 32
DISCLAIMER = (
    "Research and educational use only. NeuroLens is not a medical device and its output is "
    "not a medical diagnosis. Image-level classification does not localise or segment tumours, "
    "and model scores are not clinically validated."
)
INTERPRETATION_NOTE = (
    "Grad-CAM highlights regions that most increased the predicted class score in this model's "
    "computation. It does not establish medical correctness, is not a segmentation, and does "
    "not prove that a tumour has been localised."
)

log = logging.getLogger("neurolens.web")


class Engine:
    """Loaded model bundle. Created once per serverless instance."""

    def __init__(self, model_dir: Path) -> None:
        import onnxruntime as ort

        self.meta: dict[str, Any] = json.loads((model_dir / "meta.json").read_text("utf-8"))
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        self.cam_weights = np.load(model_dir / "cam_weights.npy")
        self.colormap = np.load(model_dir / "colormap.npy")
        pre = self.meta["preprocessing"]
        self.size = int(pre["image_size"])
        self.mean = np.asarray(pre["mean"], dtype=np.float32)[:, None, None]
        self.std = np.asarray(pre["std"], dtype=np.float32)[:, None, None]
        self.temperature: float | None = self.meta.get("temperature")
        self.policy: dict[str, Any] = self.meta["abstention"]


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    if not (MODEL_DIR / "model.onnx").is_file():
        raise HTTPException(
            503,
            "Model bundle not found. Run `neurolens export-web --run <run>` and deploy the "
            "generated deploy/vercel/model directory.",
        )
    return Engine(MODEL_DIR)


# ---------------------------------------------------------------- image handling
def decode_image(data: bytes) -> Image.Image:
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File exceeds the {MAX_UPLOAD_BYTES / 1024 / 1024:.1f} MB limit.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                fmt, (width, height) = probe.format, probe.size
                probe.verify()
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise HTTPException(413, "The image dimensions are too large.") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise HTTPException(400, "The file could not be decoded as an image.") from exc
    if fmt not in SUPPORTED_FORMATS:
        raise HTTPException(415, f"Unsupported image format '{fmt}'. Use JPEG, PNG, BMP or TIFF.")
    if width * height > MAX_PIXELS:
        raise HTTPException(413, "The image dimensions are too large.")
    if min(width, height) < MIN_SIDE:
        raise HTTPException(400, f"The image is too small ({width}x{height}).")
    try:
        with Image.open(io.BytesIO(data)) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode in {"I;16", "I;16B", "I;16L", "I", "F"}:
                array = np.asarray(img, dtype=np.float32)
                lo, hi = float(array.min()), float(array.max())
                scaled = (array - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(array)
                img = Image.fromarray(scaled.astype(np.uint8), mode="L")
            return img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(400, "The image data is truncated or corrupt.") from exc


def preprocess(image: Image.Image, engine: Engine) -> tuple[np.ndarray, np.ndarray]:
    """Return the normalised ``(1, 3, S, S)`` input and the resized RGB image for display."""
    resized = np.asarray(image.resize((engine.size, engine.size), Image.BILINEAR), dtype=np.uint8)
    chw = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
    return ((chw - engine.mean) / engine.std)[None].astype(np.float32), resized


# ------------------------------------------------------------------- numerics
def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits.astype(np.float64) / temperature
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


def resize_bilinear(array: np.ndarray, size: int) -> np.ndarray:
    """Bilinear resize matching ``torch.nn.functional.interpolate(align_corners=False)``."""

    def axis(n_in: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        src = np.clip((np.arange(size) + 0.5) * (n_in / size) - 0.5, 0, n_in - 1)
        lo = np.floor(src).astype(int)
        return lo, np.minimum(lo + 1, n_in - 1), src - lo

    y0, y1, ty = axis(array.shape[0])
    x0, x1, tx = axis(array.shape[1])
    top = array[y0][:, x0] * (1 - tx) + array[y0][:, x1] * tx
    bottom = array[y1][:, x0] * (1 - tx) + array[y1][:, x1] * tx
    return top * (1 - ty)[:, None] + bottom * ty[:, None]


def grad_cam(features: np.ndarray, engine: Engine, target: int) -> tuple[np.ndarray, bool]:
    cam = np.maximum(np.tensordot(engine.cam_weights[target], features[0], axes=(0, 0)), 0.0)
    cam = resize_bilinear(cam.astype(np.float32), engine.size)
    peak = float(cam.max())
    if peak <= 0 or not np.isfinite(peak):
        return np.zeros_like(cam), True
    return (cam - cam.min()) / (peak - cam.min() + 1e-12), False


def png_b64(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def analyze_bytes(data: bytes, engine: Engine) -> dict[str, Any]:
    image = decode_image(data)
    inputs, display = preprocess(image, engine)
    logits, features = engine.session.run(None, {"input": inputs})
    logits = logits[0]
    names, labels = engine.meta["class_names"], engine.meta["display_names"]
    raw = softmax(logits)
    calibrated = softmax(logits, engine.temperature) if engine.temperature else None
    used = calibrated if calibrated is not None else raw
    entropy = float(-(np.clip(used, 1e-12, 1) * np.log(np.clip(used, 1e-12, 1))).sum())
    norm_entropy = entropy / float(np.log(len(used)))
    reasons: list[str] = []
    policy = engine.policy
    if policy.get("enabled", True):
        if used.max() < policy["min_confidence"]:
            reasons.append(
                f"top-class probability {used.max():.2f} is below the configured threshold "
                f"{policy['min_confidence']:.2f}"
            )
        limit = policy.get("max_normalized_entropy")
        if limit is not None and norm_entropy > limit:
            reasons.append(
                f"normalised entropy {norm_entropy:.2f} exceeds the configured limit {limit:.2f}"
            )
    idx = int(raw.argmax())
    cam, degenerate = grad_cam(features, engine, idx)
    heatmap = engine.colormap[np.clip(np.rint(cam * 255), 0, 255).astype(np.uint8)]
    overlay = np.clip(0.55 * display + 0.45 * heatmap, 0, 255).astype(np.uint8)
    warnings_list = []
    if calibrated is None:
        warnings_list.append("No fitted calibration is available; probabilities are raw scores.")
    if degenerate:
        warnings_list.append("Grad-CAM produced an all-zero map for this image.")
    return {
        "prediction": {
            "predicted_class": names[idx],
            "predicted_label": labels[idx],
            "class_names": names,
            "display_names": labels,
            "raw_probabilities": dict(zip(names, map(float, raw), strict=True)),
            "calibrated_probabilities": (
                dict(zip(names, map(float, calibrated), strict=True))
                if calibrated is not None
                else None
            ),
            "raw_confidence": float(raw.max()),
            "calibrated_confidence": float(calibrated.max()) if calibrated is not None else None,
            "uncertainty": {
                "entropy_nats": entropy,
                "normalized_entropy": norm_entropy,
                "abstained": bool(reasons),
                "reasons": reasons,
                "probabilities_used": "calibrated" if calibrated is not None else "raw",
                "policy": policy,
            },
            "model": {
                "name": engine.meta["model_name"],
                "version": engine.meta["model_version"],
                "architecture": engine.meta["architecture"],
                "benchmark": engine.meta["benchmark"],
                "temperature": engine.temperature,
            },
            "warnings": warnings_list,
            "disclaimer": DISCLAIMER,
        },
        "explanation": {
            "target_class": names[idx],
            "target_layer": engine.meta["gradcam_layer"],
            "degenerate": degenerate,
            "original_png_base64": png_b64(display),
            "heatmap_png_base64": png_b64(heatmap),
            "overlay_png_base64": png_b64(overlay),
            "interpretation_note": INTERPRETATION_NOTE,
        },
    }


# ------------------------------------------------------------------------ app
app = FastAPI(
    title="NeuroLens web demo",
    description=DISCLAIMER,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    redoc_url=None,
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    ready = (MODEL_DIR / "model.onnx").is_file()
    return {"status": "ok", "ready": ready, "message": "ready" if ready else "model bundle missing"}


@app.get("/api/info")
def info() -> dict[str, Any]:
    meta = get_engine().meta
    keys = ("model_name", "model_version", "architecture", "class_names", "display_names",
            "temperature", "abstention", "benchmark", "offline_evaluation", "gradcam_layer",
            "exported_at")  # fmt: skip
    return {
        **{k: meta.get(k) for k in keys},
        "max_upload_mb": MAX_UPLOAD_BYTES / 1024 / 1024,
        "disclaimer": DISCLAIMER,
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)) -> dict[str, Any]:
    started = time.perf_counter()
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    engine = get_engine()
    result = analyze_bytes(data, engine)
    # Never log image content or the client's filename.
    log.info("analyze bytes=%d latency_ms=%.0f", len(data), (time.perf_counter() - started) * 1000)
    return result
