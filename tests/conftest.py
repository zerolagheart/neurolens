"""Shared fixtures: tiny synthetic datasets and an untrained test checkpoint.

The checkpoint fixture holds *randomly initialised* weights and exists only to exercise the
loading/inference/API code paths in tests. It is written to a temporary directory and never to
``artifacts/``.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from neurolens.data.classes import CANONICAL_CLASSES

IMAGE_SIZE = 64


def make_image(seed: int, size: int = 48, mode: str = "L") -> Image.Image:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 255, (size, size), dtype=np.uint8)
    # A bright blob whose position depends on the seed, so images differ structurally.
    cy, cx = 8 + seed % 30, 8 + (seed * 7) % 30
    array[cy : cy + 10, cx : cx + 10] = 255
    image = Image.fromarray(array, mode="L")
    return image.convert(mode) if mode != "L" else image


def png_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """``Training``/``Testing`` with 12/4 images per class (mixed JPEG/PNG, L/RGB)."""
    root = tmp_path / "dataset"
    seed = 0
    for split, count in (("Training", 12), ("Testing", 4)):
        for cls in CANONICAL_CLASSES:
            folder = root / split / cls
            folder.mkdir(parents=True)
            for i in range(count):
                seed += 1
                img = make_image(seed, mode="RGB" if i % 2 else "L")
                ext = "jpg" if i % 3 else "png"
                img.save(folder / f"{split[:2]}-{cls}_{i}.{ext}")
    return root


@pytest.fixture
def tiny_checkpoint(tmp_path: Path) -> Path:
    """A run directory with an untrained custom_cnn checkpoint (test-only, random weights)."""
    import torch

    from neurolens.data.transforms import PreprocessConfig
    from neurolens.models.factory import build_model
    from neurolens.training.checkpoint import save_checkpoint
    from neurolens.utils import software_versions, write_json

    torch.manual_seed(0)
    run_dir = tmp_path / "runs" / "test_run"
    model = build_model("custom_cnn", len(CANONICAL_CLASSES), pretrained=False, dropout=0.0)
    metadata = {
        "architecture": "custom_cnn",
        "class_names": list(CANONICAL_CLASSES),
        "preprocessing": PreprocessConfig(image_size=IMAGE_SIZE).to_dict(),
        "model": {"dropout": 0.0},
        "benchmark": "original",
        "epoch": 1,
        "model_version": "test_run-e001",
        "val_metrics": {"val_accuracy": 0.25},
        "software": software_versions(),
        "experiment": {"data": {"root": "unused"}},
    }
    save_checkpoint(run_dir / "best.pt", {"model_state": model.state_dict(), "metadata": metadata})
    write_json(run_dir / "best_meta.json", metadata)
    return run_dir
