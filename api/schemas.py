"""Pydantic response schemas (also drive the generated OpenAPI documentation)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from neurolens import DISCLAIMER


class ErrorResponse(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    status: str = Field(description="'ok' when the process is up, regardless of model readiness")
    ready: bool = Field(description="True when at least one model is loaded for inference")
    version: str
    device: str
    models_loaded: list[str]
    default_model: str | None
    load_errors: dict[str, str] = Field(default_factory=dict)
    message: str


class ArchitectureInfo(BaseModel):
    name: str
    description: str
    supports_pretrained: bool
    gradcam_target_layer: str
    trained_runs: list[str]


class ModelInfo(BaseModel):
    name: str
    architecture: str
    model_version: str
    benchmark: str | None
    epoch: int | None
    class_names: list[str]
    loaded: bool
    has_calibration: bool
    has_test_metrics: bool
    has_onnx: bool
    checkpoint_mb: float
    val_metrics: dict[str, float]


class ModelsResponse(BaseModel):
    architectures: list[ArchitectureInfo]
    models: list[ModelInfo]
    default_model: str | None


class UncertaintyInfo(BaseModel):
    entropy_nats: float
    normalized_entropy: float = Field(description="entropy / log(K); 0 = certain, 1 = uniform")
    abstained: bool = Field(description="True when the configured acceptance criterion failed")
    reasons: list[str]
    probabilities_used: str = Field(description="'calibrated' or 'raw'")
    policy: dict[str, Any]
    top_probability: float


class ModelMeta(BaseModel):
    name: str
    version: str
    architecture: str
    benchmark: str | None
    temperature: float | None


class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_label: str
    class_names: list[str]
    raw_probabilities: dict[str, float] = Field(description="softmax of the model logits")
    calibrated_probabilities: dict[str, float] | None = Field(
        description="softmax(logits / T) with T fitted on the validation split, if available"
    )
    raw_confidence: float
    calibrated_confidence: float | None
    uncertainty: UncertaintyInfo
    model: ModelMeta
    preprocessing: dict[str, Any]
    warnings: list[str]
    disclaimer: str = DISCLAIMER


class ExplanationImages(BaseModel):
    target_class: str
    target_layer: str
    degenerate: bool
    original_png_base64: str
    heatmap_png_base64: str
    overlay_png_base64: str
    legend_png_base64: str
    legend: str
    interpretation_note: str


class ExplainResponse(BaseModel):
    prediction: PredictionResponse
    explanation: ExplanationImages


class LiveStats(BaseModel):
    started_at: str
    uptime_seconds: float
    requests: dict[str, int]
    errors: dict[str, int]
    mean_latency_ms: dict[str, float]


class MetricsResponse(BaseModel):
    offline_evaluation: list[dict[str, Any]] = Field(
        description="Saved offline test-set evaluations (not computed from live traffic)"
    )
    model_comparison: dict[str, Any] | None
    live_service: LiveStats = Field(description="Statistics of this API process since startup")
    notes: list[str]
