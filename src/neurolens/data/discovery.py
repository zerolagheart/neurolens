"""Discover and validate the ``Training`` / ``Testing`` directory layout of the dataset.

The Kaggle archive is usually extracted as ``<root>/Training/<class>/*.jpg`` and
``<root>/Testing/<class>/*.jpg``, but some downloads add an extra wrapper directory. Instead of
hard-coding paths we search a couple of levels below ``root`` and validate what we find.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from neurolens.data.classes import CANONICAL_CLASSES, normalize_class_name

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
_TRAIN_NAMES = {"training", "train"}
_TEST_NAMES = {"testing", "test"}


class DatasetStructureError(RuntimeError):
    """Raised when the dataset directory does not have the expected structure."""


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    class_name: str
    source_split: str  # "train" or "test" (the dataset's own directories)


@dataclass
class DatasetLayout:
    root: Path
    train_dir: Path
    test_dir: Path | None
    class_names: tuple[str, ...]
    samples: list[Sample]
    ignored_files: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def by_split(self, split: str) -> list[Sample]:
        return [s for s in self.samples if s.source_split == split]

    def relpath(self, path: Path) -> str:
        """Path relative to the dataset root, used in reports so local paths are not exposed."""
        return Path(path).resolve().relative_to(self.root.resolve()).as_posix()


def _find_split_dir(root: Path, names: set[str], max_depth: int = 2) -> list[Path]:
    matches: list[Path] = []
    frontier = [root]
    for _ in range(max_depth + 1):
        next_frontier: list[Path] = []
        for directory in frontier:
            for child in sorted(p for p in directory.iterdir() if p.is_dir()):
                if child.name.lower() in names:
                    matches.append(child)
                else:
                    next_frontier.append(child)
        if matches:
            return matches
        frontier = next_frontier
    return matches


def _collect_split(
    split_dir: Path, split: str, class_names: tuple[str, ...], layout_warnings: list[str]
) -> tuple[list[Sample], list[Path]]:
    class_index = {name: idx for idx, name in enumerate(class_names)}
    found: dict[str, Path] = {}
    for child in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        canonical = normalize_class_name(child.name)
        if canonical is None or canonical not in class_index:
            layout_warnings.append(f"{split}: ignoring unrecognised directory '{child.name}'")
            continue
        if canonical in found:
            raise DatasetStructureError(
                f"{split}: directories '{found[canonical].name}' and '{child.name}' both map to "
                f"class '{canonical}'"
            )
        found[canonical] = child

    missing = [name for name in class_names if name not in found]
    if missing:
        raise DatasetStructureError(
            f"{split} directory '{split_dir.name}' is missing class folders: {missing}. "
            f"Expected one sub-directory per class: {list(class_names)}"
        )

    samples: list[Sample] = []
    ignored: list[Path] = []
    for name in class_names:
        class_files = sorted(p for p in found[name].rglob("*") if p.is_file())
        images = [p for p in class_files if p.suffix.lower() in IMAGE_EXTENSIONS]
        ignored.extend(p for p in class_files if p.suffix.lower() not in IMAGE_EXTENSIONS)
        if not images:
            raise DatasetStructureError(f"{split}/{found[name].name} contains no image files")
        samples.extend(Sample(p, class_index[name], name, split) for p in images)
    return samples, ignored


def discover_dataset(
    root: str | Path,
    class_names: tuple[str, ...] = CANONICAL_CLASSES,
    require_test: bool = True,
) -> DatasetLayout:
    """Locate ``Training``/``Testing`` directories under ``root`` and index every image.

    Raises:
        DatasetStructureError: if the layout is missing, ambiguous or incomplete.
    """
    root = Path(root)
    if not root.is_dir():
        raise DatasetStructureError(
            f"Dataset root '{root}' does not exist. Download the Kaggle dataset "
            "'masoudnickparvar/brain-tumor-mri-dataset' and extract it there (see README)."
        )

    train_dirs = _find_split_dir(root, _TRAIN_NAMES)
    test_dirs = _find_split_dir(root, _TEST_NAMES)
    if not train_dirs:
        raise DatasetStructureError(f"No 'Training' directory found within two levels of '{root}'")
    if len(train_dirs) > 1:
        raise DatasetStructureError(
            "Multiple candidate training directories found; point data.root at the right one: "
            + ", ".join(str(p.relative_to(root)) for p in train_dirs)
        )
    if len(test_dirs) > 1:
        raise DatasetStructureError(
            "Multiple candidate testing directories found; point data.root at the right one: "
            + ", ".join(str(p.relative_to(root)) for p in test_dirs)
        )
    if require_test and not test_dirs:
        raise DatasetStructureError(f"No 'Testing' directory found within two levels of '{root}'")

    layout_warnings: list[str] = []
    samples, ignored = _collect_split(train_dirs[0], "train", class_names, layout_warnings)
    test_dir = test_dirs[0] if test_dirs else None
    if test_dir is not None:
        test_samples, test_ignored = _collect_split(test_dir, "test", class_names, layout_warnings)
        samples.extend(test_samples)
        ignored.extend(test_ignored)

    return DatasetLayout(
        root=root,
        train_dir=train_dirs[0],
        test_dir=test_dir,
        class_names=tuple(class_names),
        samples=samples,
        ignored_files=ignored,
        warnings=layout_warnings,
    )
