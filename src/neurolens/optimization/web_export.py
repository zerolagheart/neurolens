"""Export a trained run as a PyTorch-free bundle for the serverless web demo (``deploy/vercel``).

The bundle contains an ONNX graph with two outputs, the class logits and the feature maps of
the Grad-CAM target layer, plus the weights of the final linear layer. For all supported
architectures the target layer is followed only by global average pooling, dropout (identity
at inference) and one linear layer, so

    d y_c / d A_k[i, j] = W[c, k] / (H * W)

and the Grad-CAM channel weights are exactly ``W[c, :]`` up to a positive constant that the
per-image min-max normalisation removes. The serverless function can therefore reproduce
Grad-CAM with NumPy and ONNX Runtime alone. :func:`export_web_bundle` checks this numerically
against :class:`neurolens.explainability.gradcam.GradCAM` before writing the bundle.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from torch import nn

from neurolens.data.classes import display_name
from neurolens.evaluation.calibration import AbstentionPolicy
from neurolens.evaluation.metrics import softmax
from neurolens.explainability.gradcam import GradCAM
from neurolens.models.factory import get_spec
from neurolens.optimization.onnx_export import _require_onnx, ort_session
from neurolens.training.checkpoint import model_from_checkpoint, preprocess_from_metadata
from neurolens.utils import file_size_mb, read_json, utc_now, write_json

BUNDLE_FORMAT_VERSION = 1


def split_for_cam(model: nn.Module, architecture: str) -> tuple[nn.Module, nn.Linear]:
    """Return ``(trunk, linear)``: the network up to the Grad-CAM layer, and the final layer."""
    if architecture in {"custom_cnn", "efficientnet_b0"}:
        trunk, head = model.features, model.classifier
    elif architecture == "resnet18":
        trunk = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4,
        )  # fmt: skip
        head = model.fc
    else:
        raise ValueError(f"No web export defined for architecture '{architecture}'")
    linear = head[-1]
    if not isinstance(linear, nn.Linear):
        raise ValueError(f"Expected the classifier of {architecture} to end in nn.Linear")
    return trunk, linear


class CamExportModule(nn.Module):
    """``x -> (logits, feature maps)`` with global average pooling + linear as the head."""

    def __init__(self, trunk: nn.Module, linear: nn.Linear) -> None:
        super().__init__()
        self.trunk = trunk
        self.linear = linear

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(x)
        return self.linear(features.mean(dim=(2, 3))), features


def cam_from_features(
    features: np.ndarray, weights: np.ndarray, target: int, size: int
) -> tuple[np.ndarray, bool]:
    """NumPy Grad-CAM for a GAP + linear head (mirrors the serverless implementation)."""
    cam = np.maximum(np.tensordot(weights[target], features[0], axes=(0, 0)), 0.0)
    cam = F.interpolate(
        torch.from_numpy(cam[None, None].astype(np.float32)), size=(size, size),
        mode="bilinear", align_corners=False,
    )[0, 0].numpy()  # fmt: skip
    peak = float(cam.max())
    if peak <= 0:
        return np.zeros_like(cam), True
    return (cam - cam.min()) / (peak - cam.min() + 1e-12), False


def _offline_summary(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "eval" / "test" / "metrics.json"
    if not path.exists():
        return None
    data = read_json(path)
    m = data["metrics"]
    return {
        "benchmark": data.get("benchmark"),
        "split": data["split"],
        "n_samples": m["n_samples"],
        "accuracy": m["accuracy"],
        "balanced_accuracy": m["balanced_accuracy"],
        "macro_f1": m["macro_f1"],
        "roc_auc_ovr_macro": m["roc_auc"]["macro"],
        "per_class_recall": {k: v["recall"] for k, v in m["per_class"].items()},
        "ece_raw": data["calibration"]["ece_raw"],
        "ece_calibrated": data["calibration"]["ece_calibrated"],
        "evaluated_at": data["evaluated_at"],
    }


def _parity_inputs(
    run_dir: Path, metadata: dict[str, Any], size: int, n: int, seed: int
) -> tuple[torch.Tensor, str]:
    """Up to ``n`` preprocessed validation images, or seeded noise if the dataset is absent."""
    from neurolens.data.splits import load_manifest
    from neurolens.data.transforms import load_image_rgb, preprocess_image

    root = Path(metadata.get("experiment", {}).get("data", {}).get("root", ""))
    manifest = run_dir / "manifest.csv"
    if manifest.exists() and root.is_dir():
        try:
            frame, _ = load_manifest(manifest, root)
        except FileNotFoundError:
            frame = None
        if frame is not None:
            val = frame[frame["split"] == "val"]
            val = val.sample(n=min(n, len(val)), random_state=seed)
            preprocess = preprocess_from_metadata(metadata)
            batch = [preprocess_image(load_image_rgb(p), preprocess) for p in val["path"]]
            return torch.cat(batch), f"{len(batch)} validation images"
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n, 3, size, size)).astype(np.float32)
    return torch.from_numpy(noise), f"{n} seeded random tensors (dataset unavailable)"


def export_web_bundle(
    run_dir: Path,
    out_dir: Path,
    opset: int = 17,
    prob_tolerance: float = 1e-4,
    cam_tolerance: float = 1e-3,
    n_parity: int = 8,
    seed: int = 0,
) -> dict[str, Any]:
    """Write ``model.onnx``, ``cam_weights.npy``, ``colormap.npy`` and ``meta.json``.

    Parity is checked per image on real validation images when the dataset is available:
    softmax probabilities (ONNX vs PyTorch) within ``prob_tolerance`` and the NumPy CAM within
    ``cam_tolerance`` of autograd Grad-CAM (both CAMs are min-max normalised to ``[0, 1]``).

    Raises:
        RuntimeError: if either check fails.
    """
    onnx, _ = _require_onnx()
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, metadata = model_from_checkpoint(run_dir / "best.pt", "cpu")
    architecture = metadata["architecture"]
    preprocess = preprocess_from_metadata(metadata)
    size = preprocess.image_size
    trunk, linear = split_for_cam(model, architecture)
    export_model = CamExportModule(trunk, linear).eval()

    inputs, parity_desc = _parity_inputs(run_dir, metadata, size, n_parity, seed)
    x = inputs[:1]
    with torch.inference_mode():
        reference = model(inputs).numpy()
        split_logits, _ = export_model(inputs)
    if not np.allclose(reference, split_logits.numpy(), rtol=1e-4, atol=1e-4):
        raise RuntimeError("Trunk + linear split does not reproduce the model's logits")

    onnx_path = out_dir / "model.onnx"
    kwargs: dict[str, Any] = {
        "input_names": ["input"],
        "output_names": ["logits", "features"],
        "opset_version": opset,
        "do_constant_folding": True,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    torch.onnx.export(export_model, (x,), str(onnx_path), **kwargs)
    onnx.checker.check_model(onnx.load(str(onnx_path)))

    weights = linear.weight.detach().cpu().numpy().astype(np.float32)
    np.save(out_dir / "cam_weights.npy", weights)
    lut = (colormaps["jet"](np.linspace(0.0, 1.0, 256))[:, :3] * 255).round().astype(np.uint8)
    np.save(out_dir / "colormap.npy", lut)

    # Parity on the same inputs: ONNX vs PyTorch probabilities, NumPy CAM vs autograd Grad-CAM.
    session = ort_session(onnx_path)
    prob_diffs, logit_diffs, cam_diffs, top1 = [], [], [], []
    with GradCAM.for_architecture(model, architecture) as gradcam:
        for i in range(len(inputs)):
            xi = inputs[i : i + 1]
            ort_logits, ort_features = session.run(None, {"input": xi.numpy()})
            ref_logits = reference[i : i + 1]
            prob_diffs.append(float(np.abs(softmax(ort_logits) - softmax(ref_logits)).max()))
            logit_diffs.append(float(np.abs(ort_logits - ref_logits).max()))
            top1.append(int(ort_logits.argmax()) == int(ref_logits.argmax()))
            reference_cam = gradcam(xi)
            numpy_cam, _ = cam_from_features(
                ort_features, weights, reference_cam.target_class, size
            )
            cam_diffs.append(float(np.abs(numpy_cam - reference_cam.cam).max()))
    parity = {
        "inputs": parity_desc,
        "max_abs_prob_diff": max(prob_diffs),
        "max_abs_logit_diff": max(logit_diffs),
        "max_abs_cam_diff_vs_gradcam": max(cam_diffs),
        "top1_agreement": float(np.mean(top1)),
        "prob_tolerance": prob_tolerance,
        "cam_tolerance": cam_tolerance,
    }
    if (
        parity["max_abs_prob_diff"] > prob_tolerance
        or parity["max_abs_cam_diff_vs_gradcam"] > cam_tolerance
    ):
        raise RuntimeError(f"Web bundle parity check failed: {parity}")

    calibration_path = run_dir / "calibration.json"
    temperature = None
    if calibration_path.exists():
        calibration = read_json(calibration_path)
        if calibration.get("model_version") in (None, metadata.get("model_version")):
            temperature = float(calibration["temperature"])

    meta = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "exported_at": utc_now(),
        "model_name": run_dir.name,
        "model_version": metadata.get("model_version"),
        "architecture": architecture,
        "gradcam_layer": get_spec(architecture).gradcam_layer_name,
        "cam_method": "Grad-CAM for a GAP + linear head (channel weights = final linear weights)",
        "class_names": list(metadata["class_names"]),
        "display_names": [display_name(c) for c in metadata["class_names"]],
        "preprocessing": preprocess.to_dict(),
        "temperature": temperature,
        "abstention": AbstentionPolicy().to_dict(),
        "benchmark": metadata.get("benchmark"),
        "offline_evaluation": _offline_summary(run_dir),
        "parity": parity,
        "onnx_size_mb": round(file_size_mb(onnx_path), 2),
    }
    write_json(out_dir / "meta.json", meta)
    return meta
