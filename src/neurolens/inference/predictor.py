"""Prediction pipeline: checkpoint -> preprocessing -> probabilities (+ calibration, Grad-CAM)."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from neurolens import DISCLAIMER
from neurolens.data.classes import display_name
from neurolens.data.transforms import preprocess_image, resized_display_image
from neurolens.evaluation.calibration import AbstentionPolicy, entropy, normalized_entropy
from neurolens.evaluation.metrics import softmax
from neurolens.explainability.gradcam import GradCAM
from neurolens.explainability.overlay import blend, colorize
from neurolens.training.checkpoint import model_from_checkpoint, preprocess_from_metadata
from neurolens.utils import read_json, resolve_device


@dataclass
class PredictionResult:
    predicted_class: str
    predicted_index: int
    class_names: list[str]
    raw_probabilities: list[float]
    calibrated_probabilities: list[float] | None
    temperature: float | None
    raw_confidence: float
    calibrated_confidence: float | None
    entropy: float
    normalized_entropy: float
    abstained: bool
    uncertainty_reasons: list[str]
    model_name: str
    model_version: str
    architecture: str
    benchmark: str | None
    preprocessing: dict[str, Any]
    abstention_policy: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        probs = self.calibrated_probabilities or self.raw_probabilities
        return {
            "predicted_class": self.predicted_class,
            "predicted_label": display_name(self.predicted_class),
            "class_names": self.class_names,
            "raw_probabilities": dict(zip(self.class_names, self.raw_probabilities, strict=True)),
            "calibrated_probabilities": (
                dict(zip(self.class_names, self.calibrated_probabilities, strict=True))
                if self.calibrated_probabilities is not None
                else None
            ),
            "raw_confidence": self.raw_confidence,
            "calibrated_confidence": self.calibrated_confidence,
            "uncertainty": {
                "entropy_nats": self.entropy,
                "normalized_entropy": self.normalized_entropy,
                "abstained": self.abstained,
                "reasons": self.uncertainty_reasons,
                "probabilities_used": "calibrated" if self.calibrated_probabilities else "raw",
                "policy": self.abstention_policy,
                "top_probability": float(max(probs)),
            },
            "model": {
                "name": self.model_name,
                "version": self.model_version,
                "architecture": self.architecture,
                "benchmark": self.benchmark,
                "temperature": self.temperature,
            },
            "preprocessing": self.preprocessing,
            "warnings": self.warnings,
            "disclaimer": self.disclaimer,
        }


@dataclass
class Explanation:
    original: np.ndarray  # model-resolution RGB uint8
    heatmap: np.ndarray
    overlay: np.ndarray
    cam: np.ndarray
    target_class: str
    degenerate: bool
    target_layer: str


class Predictor:
    """Holds one loaded model. Thread-safe for concurrent requests (a lock serialises calls)."""

    def __init__(
        self,
        checkpoint: Path,
        name: str | None = None,
        device: str = "auto",
        abstention: AbstentionPolicy | None = None,
        calibration_path: Path | None = None,
    ) -> None:
        self.device = resolve_device(device)
        self.checkpoint = Path(checkpoint)
        self.model, self.metadata = model_from_checkpoint(self.checkpoint, self.device)
        self.name = name or self.checkpoint.parent.name
        self.architecture: str = self.metadata["architecture"]
        self.class_names: list[str] = list(self.metadata["class_names"])
        self.preprocess = preprocess_from_metadata(self.metadata)
        self.abstention = abstention or AbstentionPolicy()
        self.temperature: float | None = None
        calibration_path = calibration_path or self.checkpoint.parent / "calibration.json"
        if calibration_path.exists():
            calibration = read_json(calibration_path)
            if calibration.get("model_version") in (None, self.metadata.get("model_version")):
                self.temperature = float(calibration["temperature"])
        self._lock = threading.Lock()

    @classmethod
    def from_run_dir(cls, run_dir: Path, **kwargs: Any) -> Predictor:
        return cls(Path(run_dir) / "best.pt", name=Path(run_dir).name, **kwargs)

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_version", self.name))

    def _result_from_logits(self, logits: np.ndarray) -> PredictionResult:
        raw = softmax(logits[None, :])[0]
        calibrated = softmax(logits[None, :], self.temperature)[0] if self.temperature else None
        used = calibrated if calibrated is not None else raw
        abstained, reasons = self.abstention.evaluate(used)
        idx = int(raw.argmax())
        warnings = []
        if calibrated is None:
            warnings.append(
                "No fitted calibration is available for this model; probabilities are raw "
                "softmax scores and may be over- or under-confident."
            )
        return PredictionResult(
            predicted_class=self.class_names[idx],
            predicted_index=idx,
            class_names=self.class_names,
            raw_probabilities=[float(p) for p in raw],
            calibrated_probabilities=[float(p) for p in calibrated]
            if calibrated is not None
            else None,
            temperature=self.temperature,
            raw_confidence=float(raw.max()),
            calibrated_confidence=float(calibrated.max()) if calibrated is not None else None,
            entropy=float(entropy(used)),
            normalized_entropy=float(normalized_entropy(used)),
            abstained=abstained,
            uncertainty_reasons=reasons,
            model_name=self.name,
            model_version=self.version,
            architecture=self.architecture,
            benchmark=self.metadata.get("benchmark"),
            preprocessing=self.preprocess.to_dict(),
            abstention_policy=self.abstention.to_dict(),
            warnings=warnings,
        )

    def predict(self, image: Image.Image) -> PredictionResult:
        tensor = preprocess_image(image, self.preprocess).to(self.device)
        with self._lock, torch.inference_mode():
            logits = self.model(tensor)[0].float().cpu().numpy()
        return self._result_from_logits(logits)

    def explain(
        self, image: Image.Image, alpha: float = 0.45
    ) -> tuple[PredictionResult, Explanation]:
        """Prediction plus Grad-CAM for the predicted class, from the same forward pass."""
        from neurolens.models.factory import get_spec

        tensor = preprocess_image(image, self.preprocess).to(self.device)
        with self._lock, GradCAM.for_architecture(self.model, self.architecture) as cam:
            cam_result = cam(tensor)
        result = self._result_from_logits(cam_result.logits.astype(np.float64))
        original = resized_display_image(image, self.preprocess)
        heatmap = colorize(cam_result.cam)
        explanation = Explanation(
            original=original,
            heatmap=heatmap,
            overlay=blend(original, heatmap, alpha),
            cam=cam_result.cam,
            target_class=self.class_names[cam_result.target_class],
            degenerate=cam_result.degenerate,
            target_layer=get_spec(self.architecture).gradcam_layer_name,
        )
        if cam_result.degenerate:
            result.warnings.append(
                "Grad-CAM produced an all-zero map for this image (no positive gradient-weighted "
                "evidence at the target layer)."
            )
        return result, explanation
