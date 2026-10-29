"""Service settings read from environment variables (see ``.env.example``)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _optional_float(name: str, default: float | None) -> float | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return None if raw.strip().lower() in {"", "none", "off"} else float(raw)


@dataclass
class Settings:
    runs_dir: Path = Path("artifacts/runs")
    reports_dir: Path = Path("artifacts/reports")
    default_model: str | None = None
    device: str = "cpu"
    max_upload_mb: float = 10.0
    max_models: int = 4
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:8501"])
    abstention_enabled: bool = True
    min_confidence: float = 0.70
    max_normalized_entropy: float | None = 0.60
    log_level: str = "INFO"

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @classmethod
    def from_env(cls) -> Settings:
        origins = os.environ.get("NEUROLENS_CORS_ORIGINS", "http://localhost:8501")
        return cls(
            runs_dir=Path(os.environ.get("NEUROLENS_RUNS_DIR", "artifacts/runs")),
            reports_dir=Path(os.environ.get("NEUROLENS_REPORTS_DIR", "artifacts/reports")),
            default_model=os.environ.get("NEUROLENS_DEFAULT_MODEL") or None,
            device=os.environ.get("NEUROLENS_DEVICE", "cpu"),
            max_upload_mb=float(os.environ.get("NEUROLENS_MAX_UPLOAD_MB", "10")),
            max_models=int(os.environ.get("NEUROLENS_MAX_MODELS", "4")),
            cors_origins=[o.strip() for o in origins.split(",") if o.strip()],
            abstention_enabled=_bool("NEUROLENS_ABSTENTION_ENABLED", True),
            min_confidence=float(os.environ.get("NEUROLENS_MIN_CONFIDENCE", "0.70")),
            max_normalized_entropy=_optional_float("NEUROLENS_MAX_NORMALIZED_ENTROPY", 0.60),
            log_level=os.environ.get("NEUROLENS_LOG_LEVEL", "INFO"),
        )
