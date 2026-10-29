"""Model service: discovers runs, loads predictors once at startup, records live statistics.

The service is shared by the FastAPI app and (in-process mode) by the Streamlit frontend so both
produce identical responses.
"""

from __future__ import annotations

import base64
import threading
import time
from collections import defaultdict
from typing import Any

from api.schemas import (
    ArchitectureInfo,
    ExplainResponse,
    ExplanationImages,
    HealthResponse,
    LiveStats,
    MetricsResponse,
    ModelInfo,
    ModelsResponse,
    PredictionResponse,
)
from api.settings import Settings
from neurolens import __version__
from neurolens.evaluation.calibration import AbstentionPolicy
from neurolens.explainability.overlay import LEGEND_TEXT, legend_image, to_png_bytes
from neurolens.inference.image_io import decode_image
from neurolens.inference.predictor import Predictor
from neurolens.inference.registry import ModelEntry, discover_models
from neurolens.models.factory import ARCHITECTURES
from neurolens.training.checkpoint import CheckpointError
from neurolens.utils import get_logger, read_json, utc_now

log = get_logger("api")

INTERPRETATION_NOTE = (
    "Grad-CAM highlights regions that most increased the predicted class score in this model's "
    "computation. It does not establish medical correctness, is not a segmentation, and does "
    "not prove that a tumour has been localised."
)


class ModelNotReadyError(RuntimeError):
    pass


class UnknownModelError(KeyError):
    pass


def _b64(array: Any) -> str:
    return base64.b64encode(to_png_bytes(array)).decode("ascii")


class ModelService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.entries: dict[str, ModelEntry] = {}
        self.predictors: dict[str, Predictor] = {}
        self.load_errors: dict[str, str] = {}
        self.default_model: str | None = None
        self.started_at = utc_now()
        self._t0 = time.monotonic()
        self._stats_lock = threading.Lock()
        self._requests: dict[str, int] = defaultdict(int)
        self._errors: dict[str, int] = defaultdict(int)
        self._latency_total: dict[str, float] = defaultdict(float)
        self._legend_b64 = _b64(legend_image())

    # ----------------------------------------------------------------- loading
    def load(self) -> None:
        """Discover runs and load up to ``max_models`` predictors (default model first)."""
        self.entries = {e.name: e for e in discover_models(self.settings.runs_dir)}
        policy = AbstentionPolicy(
            enabled=self.settings.abstention_enabled,
            min_confidence=self.settings.min_confidence,
            max_normalized_entropy=self.settings.max_normalized_entropy,
        )
        names = sorted(self.entries)
        preferred = self.settings.default_model
        if preferred and preferred not in self.entries:
            self.load_errors[preferred] = "configured default model not found in the runs directory"
        elif preferred:
            names.remove(preferred)
            names.insert(0, preferred)
        for name in names[: self.settings.max_models]:
            try:
                self.predictors[name] = Predictor(
                    self.entries[name].checkpoint,
                    name=name,
                    device=self.settings.device,
                    abstention=policy,
                )
                log.info("Loaded model %s", name)
            except (CheckpointError, RuntimeError, ValueError) as exc:
                self.load_errors[name] = str(exc)
                log.error("Failed to load model %s: %s", name, exc)
        if preferred in self.predictors:
            self.default_model = preferred
        elif self.predictors:
            self.default_model = next(iter(self.predictors))
        if not self.predictors:
            log.warning(
                "No trained model available in %s; inference endpoints return 503",
                self.settings.runs_dir.as_posix(),
            )

    @property
    def ready(self) -> bool:
        return bool(self.predictors)

    def get(self, name: str | None) -> Predictor:
        if not self.predictors:
            raise ModelNotReadyError(
                "No trained model is loaded. Train one (e.g. `neurolens train --config "
                "configs/resnet18.yaml`) and restart the service."
            )
        key = name or self.default_model
        if key not in self.predictors:
            raise UnknownModelError(
                f"Model '{key}' is not available. Loaded models: {sorted(self.predictors)}"
            )
        return self.predictors[key]

    # --------------------------------------------------------------- statistics
    def record(self, endpoint: str, seconds: float, ok: bool) -> None:
        with self._stats_lock:
            self._requests[endpoint] += 1
            self._latency_total[endpoint] += seconds
            if not ok:
                self._errors[endpoint] += 1

    def live_stats(self) -> LiveStats:
        with self._stats_lock:
            return LiveStats(
                started_at=self.started_at,
                uptime_seconds=round(time.monotonic() - self._t0, 1),
                requests=dict(self._requests),
                errors=dict(self._errors),
                mean_latency_ms={
                    k: round(1000 * v / self._requests[k], 2)
                    for k, v in self._latency_total.items()
                },
            )

    # ---------------------------------------------------------------- responses
    def health(self) -> HealthResponse:
        return HealthResponse(
            status="ok",
            ready=self.ready,
            version=__version__,
            device=self.settings.device,
            models_loaded=sorted(self.predictors),
            default_model=self.default_model,
            load_errors=self.load_errors,
            message="ready for inference" if self.ready else "no trained model loaded",
        )

    def models(self) -> ModelsResponse:
        runs_by_arch: dict[str, list[str]] = defaultdict(list)
        for entry in self.entries.values():
            runs_by_arch[entry.architecture].append(entry.name)
        return ModelsResponse(
            architectures=[
                ArchitectureInfo(
                    name=spec.name,
                    description=spec.description,
                    supports_pretrained=spec.supports_pretrained,
                    gradcam_target_layer=spec.gradcam_layer_name,
                    trained_runs=sorted(runs_by_arch.get(spec.name, [])),
                )
                for spec in ARCHITECTURES.values()
            ],
            models=[
                ModelInfo(**entry.public_dict(), loaded=entry.name in self.predictors)
                for entry in self.entries.values()
            ],
            default_model=self.default_model,
        )

    def predict(self, data: bytes, model: str | None) -> PredictionResponse:
        predictor = self.get(model)
        image = decode_image(data, self.settings.max_upload_bytes)
        return PredictionResponse(**predictor.predict(image).to_dict())

    def explain(self, data: bytes, model: str | None) -> ExplainResponse:
        predictor = self.get(model)
        image = decode_image(data, self.settings.max_upload_bytes)
        result, explanation = predictor.explain(image)
        return ExplainResponse(
            prediction=PredictionResponse(**result.to_dict()),
            explanation=ExplanationImages(
                target_class=explanation.target_class,
                target_layer=explanation.target_layer,
                degenerate=explanation.degenerate,
                original_png_base64=_b64(explanation.original),
                heatmap_png_base64=_b64(explanation.heatmap),
                overlay_png_base64=_b64(explanation.overlay),
                legend_png_base64=self._legend_b64,
                legend=LEGEND_TEXT,
                interpretation_note=INTERPRETATION_NOTE,
            ),
        )

    def metrics(self) -> MetricsResponse:
        offline = []
        for entry in self.entries.values():
            path = entry.run_dir / "eval" / "test" / "metrics.json"
            if not path.is_file():
                offline.append({"run": entry.name, "status": "not evaluated"})
                continue
            data = read_json(path)
            m = data["metrics"]
            offline.append(
                {
                    "run": entry.name,
                    "status": "evaluated",
                    "architecture": data["architecture"],
                    "benchmark": data.get("benchmark"),
                    "split": data["split"],
                    "evaluated_at": data["evaluated_at"],
                    "n_samples": m["n_samples"],
                    "accuracy": m["accuracy"],
                    "balanced_accuracy": m["balanced_accuracy"],
                    "macro_f1": m["macro_f1"],
                    "roc_auc_ovr_macro": m["roc_auc"]["macro"],
                    "per_class": m["per_class"],
                    "confusion_matrix": m["confusion_matrix"],
                    "ece_raw": data["calibration"]["ece_raw"],
                    "ece_calibrated": data["calibration"]["ece_calibrated"],
                }
            )
        comparison_path = self.settings.reports_dir / "model_comparison.json"
        return MetricsResponse(
            offline_evaluation=offline,
            model_comparison=read_json(comparison_path) if comparison_path.is_file() else None,
            live_service=self.live_stats(),
            notes=[
                "offline_evaluation comes from saved evaluation files produced by "
                "`neurolens evaluate`; it is not derived from live traffic.",
                "live_service counts requests handled by this process since it started; the "
                "service never sees ground-truth labels, so it reports no accuracy.",
            ],
        )
