"""PyTorch dataset over a list of ``(path, label)`` pairs with explicit corrupt-image handling."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from neurolens.data.transforms import load_image_rgb


class CorruptImageError(RuntimeError):
    """Raised when an image listed in a split cannot be decoded."""


def check_image(path: Path) -> str | None:
    """Return ``None`` if ``path`` decodes as an image, otherwise a short error description."""
    try:
        with Image.open(path) as img:
            img.verify()
        # ``verify`` does not decode pixel data; a full load catches truncated files.
        with Image.open(path) as img:
            img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def filter_readable(
    items: Sequence[tuple[Path, int]],
) -> tuple[list[tuple[Path, int]], list[tuple[Path, str]]]:
    """Split ``items`` into readable samples and ``(path, error)`` pairs.

    Unreadable files are returned (never silently dropped) so callers can report them.
    """
    good: list[tuple[Path, int]] = []
    bad: list[tuple[Path, str]] = []
    for path, label in items:
        error = check_image(path)
        if error is None:
            good.append((path, label))
        else:
            bad.append((path, error))
    return good, bad


class MRIDataset(Dataset):
    """Images and integer labels. Decoding failures raise :class:`CorruptImageError`."""

    def __init__(
        self,
        items: Sequence[tuple[Path, int]],
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.items = list(items)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.items[index]
        try:
            image = load_image_rgb(path)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise CorruptImageError(f"Could not decode image '{Path(path).name}': {exc}") from exc
        tensor = self.transform(image) if self.transform else image
        return tensor, label

    @property
    def labels(self) -> list[int]:
        return [label for _, label in self.items]
