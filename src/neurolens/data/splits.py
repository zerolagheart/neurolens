"""Reproducible train/validation/test manifests.

Two benchmarks are supported and always labelled in outputs:

``original``
    The dataset's own ``Testing`` directory is the test set, untouched. A stratified validation
    split is carved out of ``Training`` with a fixed seed. Unreadable files are excluded and listed.

``leakage_audited``
    Requires a prior dataset audit. Exact/near-duplicate groups are kept on one side of the
    train/validation boundary (group-aware stratified split), test images whose duplicate group
    also contains a training image are excluded, and only one image per duplicate group is kept
    inside the test set. This is a *modified* benchmark; its numbers are not comparable with
    results reported on the original split.

The test set is never used for model selection, early stopping or calibration.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from neurolens.data.dataset import check_image
from neurolens.data.discovery import DatasetLayout
from neurolens.utils import read_json, utc_now, write_json

BENCHMARKS = ("original", "leakage_audited")
MANIFEST_COLUMNS = ["relpath", "label", "class_name", "source_split", "split", "group_id"]


def stratified_split(
    labels: list[int], val_fraction: float, seed: int, groups: list[str] | None = None
) -> tuple[list[int], list[int]]:
    """Return ``(train_indices, val_indices)``; deterministic for a given seed.

    With ``groups``, no group is split across the two sides (approximate stratification via
    :class:`~sklearn.model_selection.StratifiedGroupKFold`).
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    indices = list(range(len(labels)))
    if groups is None:
        train_idx, val_idx = train_test_split(
            indices, test_size=val_fraction, stratify=labels, random_state=seed
        )
        return sorted(train_idx), sorted(val_idx)
    n_splits = max(2, round(1.0 / val_fraction))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(splitter.split(indices, labels, groups))
    return sorted(train_idx.tolist()), sorted(val_idx.tolist())


def _readability(layout: DatasetLayout, unreadable: Iterable[str] | None) -> dict[str, str]:
    if unreadable is not None:
        return dict.fromkeys(unreadable, "unreadable (from audit)")
    bad: dict[str, str] = {}
    for sample in layout.samples:
        error = check_image(sample.path)
        if error:
            bad[layout.relpath(sample.path)] = error
    return bad


def build_manifest(
    layout: DatasetLayout,
    val_fraction: float = 0.15,
    seed: int = 42,
    benchmark: str = "original",
    dup_groups: dict[str, str] | None = None,
    unreadable: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a manifest DataFrame and its metadata.

    Args:
        dup_groups: ``relpath -> duplicate group`` from the audit; required for
            ``leakage_audited``.
        unreadable: relpaths known to be unreadable (skips re-decoding every file).
    """
    if benchmark not in BENCHMARKS:
        raise ValueError(f"benchmark must be one of {BENCHMARKS}")
    if benchmark == "leakage_audited" and dup_groups is None:
        raise ValueError(
            "The leakage_audited benchmark needs duplicate groups from a dataset audit. "
            "Run `neurolens audit` first."
        )

    excluded = _readability(layout, unreadable)
    rows = []
    for sample in layout.samples:
        relpath = layout.relpath(sample.path)
        if relpath in excluded:
            continue
        group = dup_groups.get(relpath, relpath) if dup_groups else relpath
        rows.append(
            {
                "relpath": relpath,
                "label": sample.label,
                "class_name": sample.class_name,
                "source_split": sample.source_split,
                "split": sample.source_split,
                "group_id": group,
            }
        )
    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    exclusions = [{"relpath": r, "reason": reason} for r, reason in sorted(excluded.items())]

    train_mask = frame["source_split"] == "train"
    train_frame = frame[train_mask]
    groups = train_frame["group_id"].tolist() if benchmark == "leakage_audited" else None
    _, val_pos = stratified_split(train_frame["label"].tolist(), val_fraction, seed, groups)
    frame.loc[train_frame.index[val_pos], "split"] = "val"

    if benchmark == "leakage_audited":
        train_groups = set(frame.loc[frame["split"].isin(["train", "val"]), "group_id"])
        test = frame[frame["split"] == "test"]
        leaked = test[test["group_id"].isin(train_groups)]
        within_dupes = test[~test.index.isin(leaked.index)].duplicated("group_id", keep="first")
        within = test[~test.index.isin(leaked.index)][within_dupes]
        exclusions += [
            {"relpath": r, "reason": "test image duplicates a training/validation image"}
            for r in leaked["relpath"]
        ]
        exclusions += [
            {"relpath": r, "reason": "additional copy within the test duplicate group"}
            for r in within["relpath"]
        ]
        frame = frame.drop(index=leaked.index.union(within.index)).reset_index(drop=True)

    counts = frame.groupby(["split", "class_name"]).size().unstack(fill_value=0)
    metadata = {
        "benchmark": benchmark,
        "created_at": utc_now(),
        "seed": seed,
        "val_fraction": val_fraction,
        "dataset_root_name": layout.root.name,
        "class_names": list(layout.class_names),
        "counts": {s: {c: int(n) for c, n in row.items()} for s, row in counts.iterrows()},
        "excluded": exclusions,
        "procedure": (
            "Stratified random validation split of Training"
            if benchmark == "original"
            else "Group-aware stratified validation split; test filtered for cross-split and "
            "within-test duplicates"
        ),
    }
    return frame, metadata


def manifest_dir(splits_dir: Path, benchmark: str, seed: int, val_fraction: float) -> Path:
    return Path(splits_dir) / f"{benchmark}_seed{seed}_val{round(val_fraction * 100):02d}"


def save_manifest(frame: pd.DataFrame, metadata: dict[str, Any], out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "manifest.csv", index=False)
    write_json(out_dir / "manifest_meta.json", metadata)
    return out_dir / "manifest.csv"


def load_manifest(path: Path, data_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a manifest and resolve file paths against ``data_root``; validates existence."""
    path = Path(path)
    frame = pd.read_csv(path)
    missing_cols = set(MANIFEST_COLUMNS) - set(frame.columns)
    if missing_cols:
        raise ValueError(f"Manifest {path.name} is missing columns {sorted(missing_cols)}")
    meta_path = path.with_name("manifest_meta.json")
    metadata = read_json(meta_path) if meta_path.exists() else {}
    frame["path"] = [Path(data_root) / rel for rel in frame["relpath"]]
    missing = [rel for rel, p in zip(frame["relpath"], frame["path"], strict=True) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} manifest files are missing under the data root "
            f"(first: {missing[0]}). Is data.root correct?"
        )
    return frame, metadata


def split_items(frame: pd.DataFrame, split: str) -> list[tuple[Path, int]]:
    subset = frame[frame["split"] == split]
    return list(zip(subset["path"], subset["label"].astype(int), strict=True))
