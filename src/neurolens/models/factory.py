"""Model registry: builds architectures and knows each one's Grad-CAM target layer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torch import nn
from torchvision import models as tvm

from neurolens.models.custom_cnn import NeuroCNN


@dataclass(frozen=True)
class ArchitectureSpec:
    name: str
    description: str
    supports_pretrained: bool
    gradcam_layer: Callable[[nn.Module], nn.Module]
    gradcam_layer_name: str


def _build_custom_cnn(num_classes: int, pretrained: bool, dropout: float) -> nn.Module:
    if pretrained:
        raise ValueError("custom_cnn is trained from scratch; set model.pretrained=false")
    return NeuroCNN(num_classes=num_classes, dropout=dropout)


def _build_resnet18(num_classes: int, pretrained: bool, dropout: float) -> nn.Module:
    weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.resnet18(weights=weights)
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    return model


def _build_efficientnet_b0(num_classes: int, pretrained: bool, dropout: float) -> nn.Module:
    weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return model


_BUILDERS: dict[str, Callable[[int, bool, float], nn.Module]] = {
    "custom_cnn": _build_custom_cnn,
    "resnet18": _build_resnet18,
    "efficientnet_b0": _build_efficientnet_b0,
}

ARCHITECTURES: dict[str, ArchitectureSpec] = {
    "custom_cnn": ArchitectureSpec(
        "custom_cnn",
        "Compact 5-stage CNN trained from scratch (~1.2M parameters)",
        supports_pretrained=False,
        gradcam_layer=lambda m: m.features[-1],
        gradcam_layer_name="features[-1]",
    ),
    "resnet18": ArchitectureSpec(
        "resnet18",
        "ResNet-18 fine-tuned from ImageNet weights (~11.2M parameters)",
        supports_pretrained=True,
        gradcam_layer=lambda m: m.layer4,
        gradcam_layer_name="layer4",
    ),
    "efficientnet_b0": ArchitectureSpec(
        "efficientnet_b0",
        "EfficientNet-B0 fine-tuned from ImageNet weights (~4.0M parameters)",
        supports_pretrained=True,
        gradcam_layer=lambda m: m.features[-1],
        gradcam_layer_name="features[-1]",
    ),
}


class UnsupportedArchitectureError(ValueError):
    pass


def get_spec(architecture: str) -> ArchitectureSpec:
    try:
        return ARCHITECTURES[architecture]
    except KeyError as exc:
        raise UnsupportedArchitectureError(
            f"Unknown architecture '{architecture}'. Supported: {sorted(ARCHITECTURES)}"
        ) from exc


def build_model(
    architecture: str,
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.2,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Construct a model. ``pretrained=True`` downloads torchvision ImageNet weights once.

    ``freeze_backbone`` freezes everything except the classification head (linear probing).
    """
    get_spec(architecture)
    try:
        model = _BUILDERS[architecture](num_classes, pretrained, dropout)
    except (OSError, RuntimeError) as exc:
        if not pretrained:
            raise
        raise RuntimeError(
            f"Could not load ImageNet weights for {architecture} ({exc}). Check your internet "
            "connection or torch hub cache, or set model.pretrained=false."
        ) from exc
    if freeze_backbone:
        head = model.fc if architecture == "resnet18" else model.classifier
        for param in model.parameters():
            param.requires_grad = False
        for param in head.parameters():
            param.requires_grad = True
    return model


def gradcam_target_layer(model: nn.Module, architecture: str) -> nn.Module:
    return get_spec(architecture).gradcam_layer(model)


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
