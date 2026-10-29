"""Discover trained runs on disk without loading their weights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neurolens.utils import file_size_mb, read_json


@dataclass
class ModelEntry:
    name: str
    run_dir: Path
    architecture: str
    model_version: str
    benchmark: str | None
    epoch: int | None
    class_names: list[str]
    has_calibration: bool
    has_test_metrics: bool
    has_onnx: bool
    checkpoint_mb: float
    val_metrics: dict[str, Any]

    @property
    def checkpoint(self) -> Path:
        return self.run_dir / "best.pt"

    def public_dict(self) -> dict[str, Any]:
        """Serializable summary without filesystem paths."""
        return {
            "name": self.name,
            "architecture": self.architecture,
            "model_version": self.model_version,
            "benchmark": self.benchmark,
            "epoch": self.epoch,
            "class_names": self.class_names,
            "has_calibration": self.has_calibration,
            "has_test_metrics": self.has_test_metrics,
            "has_onnx": self.has_onnx,
            "checkpoint_mb": round(self.checkpoint_mb, 2),
            "val_metrics": self.val_metrics,
        }


def discover_models(runs_dir: Path) -> list[ModelEntry]:
    """Every run directory under ``runs_dir`` with ``best.pt`` and ``best_meta.json``."""
    entries: list[ModelEntry] = []
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return entries
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        checkpoint, meta_path = run_dir / "best.pt", run_dir / "best_meta.json"
        if not (checkpoint.is_file() and meta_path.is_file()):
            continue
        meta = read_json(meta_path)
        entries.append(
            ModelEntry(
                name=run_dir.name,
                run_dir=run_dir,
                architecture=meta.get("architecture", "unknown"),
                model_version=meta.get("model_version", run_dir.name),
                benchmark=meta.get("benchmark"),
                epoch=meta.get("epoch"),
                class_names=list(meta.get("class_names", [])),
                has_calibration=(run_dir / "calibration.json").is_file(),
                has_test_metrics=(run_dir / "eval" / "test" / "metrics.json").is_file(),
                has_onnx=(run_dir / "onnx" / "model.onnx").is_file(),
                checkpoint_mb=file_size_mb(checkpoint),
                val_metrics=meta.get("val_metrics", {}),
            )
        )
    return entries
