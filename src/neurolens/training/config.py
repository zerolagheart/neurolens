"""Experiment configuration: typed dataclasses loaded from YAML with dotted CLI overrides."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from neurolens.data.transforms import AugmentationConfig, PreprocessConfig

MONITOR_METRICS = {
    "val_loss": "min",
    "val_accuracy": "max",
    "val_balanced_accuracy": "max",
    "val_macro_f1": "max",
}


@dataclass
class DataConfig:
    root: str = "data/raw/brain-tumor-mri-dataset"
    benchmark: str = "original"  # original | leakage_audited
    val_fraction: float = 0.15
    splits_dir: str = "artifacts/splits"
    audit_dir: str = "artifacts/audit"
    manifest: str | None = None  # explicit manifest.csv; built automatically when null
    num_workers: int = 4


@dataclass
class ModelConfig:
    architecture: str = "resnet18"
    pretrained: bool = True
    dropout: float = 0.2
    freeze_backbone: bool = False


@dataclass
class TrainingConfig:
    epochs: int = 15
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"  # adamw | sgd
    scheduler: str = "cosine"  # cosine | plateau | none
    label_smoothing: float = 0.0
    monitor: str = "val_macro_f1"
    early_stopping_patience: int = 5
    amp: bool = True  # only used on CUDA
    device: str = "auto"
    deterministic: bool = False
    max_train_batches: int | None = None  # debugging / smoke tests
    max_eval_batches: int | None = None


@dataclass
class ExperimentConfig:
    experiment_name: str = "resnet18_baseline"
    seed: int = 42
    runs_dir: str = "artifacts/runs"
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessConfig = field(default_factory=PreprocessConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["preprocessing"]["mean"] = list(self.preprocessing.mean)
        data["preprocessing"]["std"] = list(self.preprocessing.std)
        data["augmentation"]["scale"] = list(self.augmentation.scale)
        return data

    @property
    def run_dir(self) -> Path:
        return Path(self.runs_dir) / self.experiment_name

    def validate(self) -> None:
        from neurolens.data.splits import BENCHMARKS
        from neurolens.models.factory import get_spec

        spec = get_spec(self.model.architecture)
        if self.model.pretrained and not spec.supports_pretrained:
            raise ValueError(f"{spec.name} has no pretrained weights; set model.pretrained=false")
        if self.data.benchmark not in BENCHMARKS:
            raise ValueError(f"data.benchmark must be one of {BENCHMARKS}")
        if self.training.monitor not in MONITOR_METRICS:
            raise ValueError(f"training.monitor must be one of {sorted(MONITOR_METRICS)}")
        if self.training.epochs < 1 or self.training.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.training.scheduler not in {"cosine", "plateau", "none"}:
            raise ValueError("training.scheduler must be cosine, plateau or none")
        if self.training.optimizer not in {"adamw", "sgd"}:
            raise ValueError("training.optimizer must be adamw or sgd")


def _build(cls: type, values: dict[str, Any]) -> Any:
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in values.items():
        default = getattr(cls(), name) if name in known else None
        if dataclasses.is_dataclass(default) and isinstance(value, dict):
            kwargs[name] = _build(type(default), value)
        elif isinstance(default, tuple) and isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _coerce(raw: str) -> Any:
    return yaml.safe_load(raw)


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``section.key=value`` strings (values parsed as YAML scalars)."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like section.key=value")
        dotted, raw = item.split("=", 1)
        target = data
        keys = dotted.strip().split(".")
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = _coerce(raw)
    return data


def load_config(
    path: str | Path | None = None, overrides: list[str] | None = None
) -> ExperimentConfig:
    data: dict[str, Any] = {}
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        data = loaded or {}
    data = apply_overrides(data, overrides or [])
    config: ExperimentConfig = _build(ExperimentConfig, data)
    config.validate()
    return config


def config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    return _build(ExperimentConfig, data)


def save_config(config: ExperimentConfig, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
