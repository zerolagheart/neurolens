from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neurolens.models.factory import (
    ARCHITECTURES,
    UnsupportedArchitectureError,
    build_model,
    count_parameters,
    gradcam_target_layer,
)
from neurolens.training.checkpoint import (
    CheckpointError,
    load_checkpoint,
    model_from_checkpoint,
    save_checkpoint,
)


@pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
def test_output_dimensions(arch: str) -> None:
    model = build_model(arch, num_classes=4, pretrained=False).eval()
    with torch.inference_mode():
        out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 4)
    probs = torch.softmax(out, dim=1)
    assert torch.allclose(probs.sum(1), torch.ones(2), atol=1e-6)
    assert isinstance(gradcam_target_layer(model, arch), torch.nn.Module)


def test_freeze_backbone_leaves_only_head_trainable() -> None:
    model = build_model("resnet18", 4, pretrained=False, freeze_backbone=True)
    counts = count_parameters(model)
    assert counts["trainable"] == 512 * 4 + 4
    assert counts["total"] > counts["trainable"]


def test_unknown_architecture() -> None:
    with pytest.raises(UnsupportedArchitectureError):
        build_model("vit_huge", 4)


def test_custom_cnn_rejects_pretrained() -> None:
    with pytest.raises(ValueError):
        build_model("custom_cnn", 4, pretrained=True)


def test_checkpoint_roundtrip_and_validation(tiny_checkpoint: Path) -> None:
    model, meta = model_from_checkpoint(tiny_checkpoint / "best.pt")
    assert meta["architecture"] == "custom_cnn"
    assert not model.training
    with pytest.raises(CheckpointError, match="architecture"):
        model_from_checkpoint(tiny_checkpoint / "best.pt", expected_architecture="resnet18")
    with pytest.raises(CheckpointError, match="classes"):
        model_from_checkpoint(tiny_checkpoint / "best.pt", expected_classes=["a", "b"])


def test_missing_and_foreign_checkpoints(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError, match="not found"):
        load_checkpoint(tmp_path / "missing.pt")
    foreign = tmp_path / "foreign.pt"
    torch.save({"weights": torch.zeros(1)}, foreign)
    with pytest.raises(CheckpointError, match="not a NeuroLens checkpoint"):
        load_checkpoint(foreign)


def test_mismatched_weights_are_rejected(tmp_path: Path) -> None:
    model = build_model("custom_cnn", 4)
    meta = {"architecture": "resnet18", "class_names": ["a", "b", "c", "d"], "preprocessing": {}}
    save_checkpoint(tmp_path / "bad.pt", {"model_state": model.state_dict(), "metadata": meta})
    with pytest.raises(CheckpointError, match="do not fit"):
        model_from_checkpoint(tmp_path / "bad.pt")
