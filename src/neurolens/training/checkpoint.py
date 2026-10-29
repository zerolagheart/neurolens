"""Checkpoint save/load with metadata validation.

Checkpoints contain only tensors and plain Python types and are always loaded with
``torch.load(weights_only=True)``, so loading never executes pickled code.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch
from torch import nn

from neurolens.data.transforms import PreprocessConfig
from neurolens.models.factory import build_model

CHECKPOINT_FORMAT_VERSION = 1


class CheckpointError(RuntimeError):
    """Missing, unreadable or incompatible checkpoint."""


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Atomically write ``state`` (write to a temp file, then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"format_version": CHECKPOINT_FORMAT_VERSION, **state}, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"Checkpoint not found: {path.name}")
    try:
        state = torch.load(path, map_location=map_location, weights_only=True)
    except (RuntimeError, OSError, EOFError, ValueError, pickle.UnpicklingError) as exc:
        raise CheckpointError(f"Could not read checkpoint {path.name}: {exc}") from exc
    if not isinstance(state, dict) or "model_state" not in state or "metadata" not in state:
        raise CheckpointError(f"{path.name} is not a NeuroLens checkpoint")
    if state.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError(
            f"{path.name} has format version {state.get('format_version')}, expected "
            f"{CHECKPOINT_FORMAT_VERSION}"
        )
    return state


def validate_metadata(
    metadata: dict[str, Any],
    expected_architecture: str | None = None,
    expected_classes: list[str] | tuple[str, ...] | None = None,
) -> None:
    for key in ("architecture", "class_names", "preprocessing"):
        if key not in metadata:
            raise CheckpointError(f"Checkpoint metadata is missing '{key}'")
    if expected_architecture and metadata["architecture"] != expected_architecture:
        raise CheckpointError(
            f"Checkpoint architecture '{metadata['architecture']}' does not match "
            f"'{expected_architecture}'"
        )
    if expected_classes is not None and list(metadata["class_names"]) != list(expected_classes):
        raise CheckpointError(
            f"Checkpoint classes {metadata['class_names']} do not match {list(expected_classes)}"
        )


def model_from_checkpoint(
    path: Path,
    device: torch.device | str = "cpu",
    expected_architecture: str | None = None,
    expected_classes: list[str] | tuple[str, ...] | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild the model described by a checkpoint and load its weights (strict)."""
    state = load_checkpoint(path, map_location="cpu")
    metadata = state["metadata"]
    validate_metadata(metadata, expected_architecture, expected_classes)
    model = build_model(
        metadata["architecture"],
        num_classes=len(metadata["class_names"]),
        pretrained=False,
        dropout=float(metadata.get("model", {}).get("dropout", 0.0)),
    )
    try:
        model.load_state_dict(state["model_state"], strict=True)
    except RuntimeError as exc:
        raise CheckpointError(
            f"Weights in {Path(path).name} do not fit the architecture: {exc}"
        ) from exc
    model.to(device).eval()
    return model, metadata


def preprocess_from_metadata(metadata: dict[str, Any]) -> PreprocessConfig:
    return PreprocessConfig.from_dict(metadata["preprocessing"])
