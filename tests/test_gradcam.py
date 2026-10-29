from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from neurolens.explainability.gradcam import GradCAM
from neurolens.explainability.overlay import blend, colorize, legend_image
from neurolens.models.factory import ARCHITECTURES, UnsupportedArchitectureError, build_model


@pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
def test_gradcam_shape_range_and_hook_cleanup(arch: str) -> None:
    torch.manual_seed(0)
    model = build_model(arch, 4, pretrained=False)
    x = torch.randn(1, 3, 64, 64)
    with GradCAM.for_architecture(model, arch) as cam:
        layer = cam.target_layer
        assert len(layer._forward_hooks) == 1
        result = cam(x)
    assert len(layer._forward_hooks) == 0
    assert result.cam.shape == (64, 64)
    assert np.isfinite(result.cam).all()
    assert result.cam.min() >= 0.0 and result.cam.max() <= 1.0 + 1e-6
    assert np.isclose(result.probabilities.sum(), 1.0, atol=1e-5)
    assert result.target_class == int(result.probabilities.argmax())
    # parameter gradients are not left behind and the original train/eval mode is restored
    assert all(p.grad is None for p in model.parameters())
    assert model.training


def test_gradcam_works_inside_inference_mode_and_with_frozen_weights() -> None:
    model = build_model("resnet18", 4, pretrained=False, freeze_backbone=True)
    for p in model.parameters():
        p.requires_grad = False
    with torch.inference_mode():
        x = torch.randn(1, 3, 64, 64)
    with GradCAM.for_architecture(model, "resnet18") as cam:
        result = cam(x, target_class=2)
    assert result.target_class == 2
    assert np.isfinite(result.cam).all()


def test_gradcam_rejects_batch_and_bad_target() -> None:
    model = build_model("custom_cnn", 4)
    with GradCAM.for_architecture(model, "custom_cnn") as cam:
        with pytest.raises(ValueError):
            cam(torch.randn(2, 3, 32, 32))
        with pytest.raises(ValueError):
            cam(torch.randn(1, 3, 32, 32), target_class=9)


def test_unsupported_architecture() -> None:
    with pytest.raises(UnsupportedArchitectureError):
        GradCAM.for_architecture(nn.Linear(3, 4), "mlp")


def test_overlay_helpers() -> None:
    cam = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
    heat = colorize(cam)
    assert heat.shape == (32, 32, 3) and heat.dtype == np.uint8
    image = np.full((32, 32, 3), 100, dtype=np.uint8)
    over = blend(image, heat, 0.5)
    assert over.shape == image.shape
    with pytest.raises(ValueError):
        blend(image[:10], heat)
    assert legend_image().ndim == 3
