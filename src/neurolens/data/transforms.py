"""Image preprocessing shared by training, evaluation and inference.

The *same* :class:`PreprocessConfig` is stored inside every checkpoint and rebuilt at inference
time, so a served model always sees inputs prepared exactly as during evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class PreprocessConfig:
    """Deterministic preprocessing: convert to RGB, resize to a square, normalise."""

    image_size: int = 224
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["mean"] = list(self.mean)
        data["std"] = list(self.std)
        data["color_mode"] = "RGB (grayscale replicated to 3 channels)"
        data["resize"] = "bilinear resize to image_size x image_size (aspect ratio not preserved)"
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreprocessConfig:
        return cls(
            image_size=int(data.get("image_size", 224)),
            mean=tuple(data.get("mean", IMAGENET_MEAN)),
            std=tuple(data.get("std", IMAGENET_STD)),
        )


@dataclass
class AugmentationConfig:
    """Training-only augmentation. Kept mild: MRI intensities and anatomy carry the signal."""

    enabled: bool = True
    hflip_p: float = 0.5
    rotation_degrees: float = 10.0
    scale: tuple[float, float] = field(default=(0.85, 1.0))
    brightness: float = 0.1
    contrast: float = 0.1


def load_image_rgb(source: str | Path | IO[bytes]) -> Image.Image:
    """Open an image and return a fully-loaded RGB copy.

    Handles grayscale (``L``), 16-bit (``I;16``/``I``), palette and alpha images, and applies the
    EXIF orientation tag if present.
    """
    with Image.open(source) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode in {"I;16", "I;16B", "I;16L", "I", "F"}:
            array = np.asarray(img, dtype=np.float32)
            lo, hi = float(array.min()), float(array.max())
            scaled = (array - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(array)
            img = Image.fromarray(scaled.astype(np.uint8), mode="L")
        return img.convert("RGB")


def build_eval_transform(cfg: PreprocessConfig) -> T.Compose:
    return T.Compose(
        [
            T.Resize((cfg.image_size, cfg.image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(cfg.mean, cfg.std),
        ]
    )


def build_train_transform(cfg: PreprocessConfig, aug: AugmentationConfig) -> T.Compose:
    if not aug.enabled:
        return build_eval_transform(cfg)
    ops: list[Any] = [
        T.RandomResizedCrop(
            (cfg.image_size, cfg.image_size), scale=tuple(aug.scale), ratio=(0.9, 1.1)
        ),
    ]
    if aug.hflip_p > 0:
        ops.append(T.RandomHorizontalFlip(p=aug.hflip_p))
    if aug.rotation_degrees > 0:
        ops.append(T.RandomRotation(aug.rotation_degrees))
    if aug.brightness > 0 or aug.contrast > 0:
        ops.append(T.ColorJitter(brightness=aug.brightness, contrast=aug.contrast))
    ops += [T.ToTensor(), T.Normalize(cfg.mean, cfg.std)]
    return T.Compose(ops)


def preprocess_image(image: Image.Image, cfg: PreprocessConfig) -> torch.Tensor:
    """Return a ``(1, 3, H, W)`` tensor ready for the model."""
    return build_eval_transform(cfg)(image.convert("RGB")).unsqueeze(0)


def resized_display_image(image: Image.Image, cfg: PreprocessConfig) -> np.ndarray:
    """The RGB image (uint8, H x W x 3) at model resolution, for overlays."""
    resized = image.convert("RGB").resize((cfg.image_size, cfg.image_size), Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)
