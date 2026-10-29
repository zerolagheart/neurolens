"""Turn a Grad-CAM map into a colour heatmap, an overlay and a legend image."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from PIL import Image

COLORMAP = "jet"
LEGEND_TEXT = (
    "Colour shows relative Grad-CAM influence on the predicted class score (blue = low, "
    "red = high), normalised per image. It reflects the model's computation, not a tumour "
    "boundary or a medical finding."
)


def colorize(cam: np.ndarray, colormap: str = COLORMAP) -> np.ndarray:
    """``(H, W)`` map in [0, 1] -> ``(H, W, 3)`` uint8 RGB."""
    rgba = colormaps[colormap](np.clip(cam, 0.0, 1.0))
    return (rgba[..., :3] * 255).astype(np.uint8)


def blend(image_rgb: np.ndarray, heatmap_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    if image_rgb.shape != heatmap_rgb.shape:
        raise ValueError(f"shape mismatch {image_rgb.shape} vs {heatmap_rgb.shape}")
    mixed = (1 - alpha) * image_rgb.astype(np.float32) + alpha * heatmap_rgb.astype(np.float32)
    return np.clip(mixed, 0, 255).astype(np.uint8)


def legend_image(
    colormap: str = COLORMAP, width_px: int = 480, background: str = "#111a2b", ink: str = "#cbd5e1"
) -> np.ndarray:
    """A horizontal colour bar labelled low -> high influence (dark background by default)."""
    fig, ax = plt.subplots(figsize=(width_px / 100, 0.75), dpi=100, facecolor=background)
    ax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=colormap)
    ax.set_yticks([])
    ax.set_xticks([0, 127.5, 255], ["low", "relative influence", "high"])
    ax.tick_params(labelsize=8, colors=ink)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.2)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor=background)
    plt.close(fig)
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("RGB"))


def to_png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()
