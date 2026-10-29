"""Canonical class names for the Nickparvar Brain Tumor MRI dataset.

Class indices follow sorted directory names (the same convention as ``torchvision``'s
``ImageFolder``), so a checkpoint's class mapping is stable across machines.
"""

from __future__ import annotations

CANONICAL_CLASSES: tuple[str, ...] = ("glioma", "meningioma", "notumor", "pituitary")

DISPLAY_NAMES: dict[str, str] = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No tumor",
    "pituitary": "Pituitary tumor",
}

# Directory spellings seen in this dataset and its predecessors (e.g. the SARTAJ dataset).
_ALIASES: dict[str, str] = {
    "glioma": "glioma",
    "glioma_tumor": "glioma",
    "meningioma": "meningioma",
    "meningioma_tumor": "meningioma",
    "notumor": "notumor",
    "no_tumor": "notumor",
    "no tumor": "notumor",
    "no-tumor": "notumor",
    "pituitary": "pituitary",
    "pituitary_tumor": "pituitary",
    "pituitary tumor": "pituitary",
}


def normalize_class_name(name: str) -> str | None:
    """Map a directory name to a canonical class, or ``None`` if it is not recognised."""
    return _ALIASES.get(name.strip().lower())


def display_name(class_name: str) -> str:
    return DISPLAY_NAMES.get(class_name, class_name)


def class_to_index(class_names: tuple[str, ...] | list[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(class_names)}
