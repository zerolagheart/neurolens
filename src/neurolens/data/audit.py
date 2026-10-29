"""Dataset integrity audit: counts, image properties, unreadable files, exact and near duplicates.

Exact duplicates are found with SHA-256 over file bytes. Near-duplicates are *candidates* found
with a 64-bit DCT perceptual hash (pHash) and a Hamming-distance threshold; they require manual
review. Nothing is deleted or moved: the audit only reports, and
:func:`neurolens.data.splits.build_manifest` can use the findings to create a separate,
clearly-labelled "leakage-audited" benchmark.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from scipy.fft import dctn

from neurolens.data.discovery import DatasetLayout
from neurolens.utils import get_logger, utc_now, write_json

log = get_logger("audit")

AUDIT_LIMITATIONS = [
    "The dataset provides no patient identifiers, so patient-level independence between "
    "partitions cannot be established. Different slices from the same patient may appear in "
    "both training and test data even when no two files are duplicates.",
    "Exact duplicates are byte-identical files (SHA-256). Re-encoded, resized or cropped copies "
    "are only caught by the perceptual hash.",
    "Near-duplicate detection uses a 64-bit DCT perceptual hash with a Hamming-distance "
    "threshold. It produces candidates, not proof: brain MRI slices share global structure, so "
    "false positives are expected at loose thresholds and subtle duplicates can be missed at "
    "strict ones. Review the reported pairs visually before drawing conclusions.",
    "The audit checks files, not labels: it cannot detect mislabelled images except where the "
    "same (or nearly the same) image appears under two different classes.",
]


@dataclass
class AuditConfig:
    near_duplicate_threshold: int = 4
    workers: int = 8
    max_examples: int = 25


def perceptual_hash(image: Image.Image) -> int:
    """64-bit DCT perceptual hash (pHash): low-frequency DCT coefficients vs. their median."""
    gray = image.convert("L").resize((32, 32), Image.LANCZOS)
    coeffs = dctn(np.asarray(gray, dtype=np.float64), norm="ortho")[:8, :8].flatten()
    median = np.median(coeffs[1:])  # exclude the DC term, which only encodes mean brightness
    bits = coeffs > median
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _popcount(values: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(values)
    as_bytes = values.view(np.uint8).reshape(*values.shape, 8)
    return np.unpackbits(as_bytes, axis=-1).sum(axis=-1)


def near_duplicate_pairs(
    hashes: np.ndarray, threshold: int, chunk: int = 512
) -> list[tuple[int, int, int]]:
    """All index pairs ``(i, j, distance)`` with ``i < j`` and Hamming distance <= threshold."""
    hashes = np.asarray(hashes, dtype=np.uint64)
    pairs: list[tuple[int, int, int]] = []
    for start in range(0, len(hashes), chunk):
        block = hashes[start : start + chunk, None] ^ hashes[None, :]
        dist = _popcount(block)
        rows, cols = np.nonzero(dist <= threshold)
        rows = rows + start
        keep = cols > rows
        pairs.extend(
            zip(rows[keep].tolist(), cols[keep].tolist(), dist[rows[keep] - start, cols[keep]].tolist(), strict=True)
        )
    return pairs


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _inspect_file(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"bytes": path.stat().st_size}
    record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        with Image.open(path) as img:
            record.update(format=img.format, mode=img.mode, width=img.width, height=img.height)
            img.load()
            record["phash"] = f"{perceptual_hash(img):016x}"  # hex: pandas cannot hold uint64+NaN
        record["error"] = None
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        record.update(error=f"{type(exc).__name__}: {exc}", phash=None)
    return record


@dataclass
class AuditResult:
    report: dict[str, Any]
    files: pd.DataFrame
    pairs: pd.DataFrame


def run_audit(
    layout: DatasetLayout,
    config: AuditConfig | None = None,
    partitions: dict[str, str] | None = None,
) -> AuditResult:
    """Audit every image in ``layout``.

    Args:
        partitions: optional ``relpath -> partition`` map (e.g. from a split manifest with
            ``train``/``val``/``test``). Defaults to the dataset's own Training/Testing split.
    """
    config = config or AuditConfig()
    samples = layout.samples
    log.info("Auditing %d files with %d workers", len(samples), config.workers)
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        inspected = list(pool.map(_inspect_file, [s.path for s in samples]))

    rows = []
    for sample, info in zip(samples, inspected, strict=True):
        relpath = layout.relpath(sample.path)
        rows.append(
            {
                "relpath": relpath,
                "source_split": sample.source_split,
                "partition": (partitions or {}).get(relpath, sample.source_split),
                "class_name": sample.class_name,
                **info,
            }
        )
    files = pd.DataFrame(rows)
    files["extension"] = files["relpath"].str.rsplit(".", n=1).str[-1].str.lower()

    readable = files[files["error"].isna()].reset_index()
    uf = _UnionFind(len(files))

    # Exact duplicates: identical SHA-256.
    pair_rows: list[dict[str, Any]] = []
    for _, group in files.groupby("sha256"):
        idx = group.index.tolist()
        for a_pos, a in enumerate(idx):
            for b in idx[a_pos + 1 :]:
                uf.union(a, b)
                pair_rows.append({"a": a, "b": b, "kind": "exact", "distance": 0})

    # Near duplicates among readable images with different bytes.
    hashes = np.array([int(h, 16) for h in readable["phash"]], dtype=np.uint64)
    for i, j, dist in near_duplicate_pairs(hashes, config.near_duplicate_threshold):
        a, b = int(readable.loc[i, "index"]), int(readable.loc[j, "index"])
        if files.at[a, "sha256"] == files.at[b, "sha256"]:
            continue
        uf.union(a, b)
        pair_rows.append({"a": a, "b": b, "kind": "near", "distance": dist})

    files["dup_group"] = [f"g{uf.find(i):05d}" for i in range(len(files))]

    pairs = pd.DataFrame(pair_rows, columns=["a", "b", "kind", "distance"])
    for side in ("a", "b"):
        for col in ("relpath", "partition", "class_name"):
            pairs[f"{col}_{side}"] = files.loc[pairs[side], col].to_numpy() if len(pairs) else []
    pairs = pairs.drop(columns=["a", "b"])
    pairs["cross_partition"] = pairs["partition_a"] != pairs["partition_b"]
    pairs["label_conflict"] = pairs["class_name_a"] != pairs["class_name_b"]

    report = _build_report(layout, files, pairs, config)
    return AuditResult(report=report, files=files, pairs=pairs)


def _partition_pair(a: str, b: str) -> str:
    return "-".join(sorted((a, b)))


def _build_report(
    layout: DatasetLayout, files: pd.DataFrame, pairs: pd.DataFrame, config: AuditConfig
) -> dict[str, Any]:
    readable = files[files["error"].isna()]
    counts = (
        files.groupby(["partition", "class_name"]).size().unstack(fill_value=0).astype(int)
    )

    def pair_summary(subset: pd.DataFrame) -> dict[str, Any]:
        summary: dict[str, Any] = {"pairs": len(subset)}
        by_partition: dict[str, int] = Counter(
            _partition_pair(a, b)
            for a, b in zip(subset["partition_a"], subset["partition_b"], strict=True)
        )
        summary["by_partition_pair"] = dict(sorted(by_partition.items()))
        summary["cross_partition_pairs"] = int(subset["cross_partition"].sum())
        summary["label_conflict_pairs"] = int(subset["label_conflict"].sum())
        return summary

    exact = pairs[pairs["kind"] == "exact"]
    near = pairs[pairs["kind"] == "near"]

    # Leakage: evaluation-partition files that have an exact/near match in a training partition.
    leak: dict[str, Any] = {}
    partitions = sorted(files["partition"].unique())
    groups = files.groupby("dup_group")["partition"].agg(set)
    group_of = files.set_index("relpath")["dup_group"]
    for target in [p for p in partitions if p != "train"]:
        target_files = files.loc[files["partition"] == target, "relpath"]
        affected = [r for r in target_files if "train" in groups[group_of[r]]]
        leak[f"{target}_files_with_match_in_train"] = len(affected)
        leak[f"{target}_fraction_with_match_in_train"] = round(len(affected) / max(len(target_files), 1), 4)

    multi = files.groupby("dup_group").size()
    examples = pairs.sort_values(["cross_partition", "distance"], ascending=[False, True])
    example_cols = ["kind", "distance", "relpath_a", "partition_a", "class_name_a",
                    "relpath_b", "partition_b", "class_name_b"]

    sizes = readable[["width", "height"]].astype(int)
    size_counts = Counter(f"{w}x{h}" for w, h in zip(sizes["width"], sizes["height"], strict=True))
    return {
        "generated_at": utc_now(),
        "dataset_root_name": layout.root.name,
        "class_names": list(layout.class_names),
        "total_files": len(files),
        "counts_by_partition_and_class": {
            part: {cls: int(v) for cls, v in row.items()} for part, row in counts.iterrows()
        },
        "counts_by_source_split": files["source_split"].value_counts().to_dict(),
        "formats": files["format"].fillna("unreadable").value_counts().to_dict(),
        "extensions": files["extension"].value_counts().to_dict(),
        "color_modes": files["mode"].fillna("unreadable").value_counts().to_dict(),
        "image_size": {
            "unique_sizes": len(size_counts),
            "most_common": [{"size": s, "count": c} for s, c in size_counts.most_common(10)],
            "width": _describe(sizes["width"]),
            "height": _describe(sizes["height"]),
        },
        "unreadable_files": files.loc[files["error"].notna(), ["relpath", "error"]].to_dict("records"),
        "ignored_non_image_files": [layout.relpath(p) for p in layout.ignored_files],
        "layout_warnings": layout.warnings,
        "exact_duplicates": {
            **pair_summary(exact),
            "files_in_duplicate_sets": int(files["sha256"].duplicated(keep=False).sum()),
            "duplicate_sets": int((files.groupby("sha256").size() > 1).sum()),
        },
        "near_duplicates": {
            **pair_summary(near),
            "method": "64-bit DCT pHash, Hamming distance",
            "threshold": config.near_duplicate_threshold,
        },
        "duplicate_groups": {
            "groups_with_multiple_files": int((multi > 1).sum()),
            "largest_group_size": int(multi.max()) if len(multi) else 0,
        },
        "leakage": leak,
        "examples": examples[example_cols].head(config.max_examples).to_dict("records"),
        "limitations": AUDIT_LIMITATIONS,
    }


def _describe(series: pd.Series) -> dict[str, float]:
    if series.empty:
        return {}
    return {
        "min": int(series.min()),
        "median": float(series.median()),
        "max": int(series.max()),
        "mean": round(float(series.mean()), 1),
    }


def save_audit(result: AuditResult, out_dir: Path) -> dict[str, Path]:
    """Write ``audit_report.json``, ``audit_files.csv`` and ``duplicate_pairs.csv``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "report": out_dir / "audit_report.json",
        "files": out_dir / "audit_files.csv",
        "pairs": out_dir / "duplicate_pairs.csv",
    }
    write_json(paths["report"], result.report)
    result.files.to_csv(paths["files"], index=False)
    result.pairs.to_csv(paths["pairs"], index=False)
    return paths


def duplicate_groups_from_csv(path: Path) -> dict[str, str]:
    """``relpath -> dup_group`` from a saved ``audit_files.csv``."""
    frame = pd.read_csv(path, usecols=["relpath", "dup_group"])
    return dict(zip(frame["relpath"], frame["dup_group"], strict=True))


def summarize_groups(groups: dict[str, str]) -> dict[str, int]:
    sizes = Counter(groups.values())
    return {"groups": len(sizes), "multi_file_groups": sum(1 for v in sizes.values() if v > 1)}


def group_members(groups: dict[str, str]) -> dict[str, list[str]]:
    members: dict[str, list[str]] = defaultdict(list)
    for relpath, group in groups.items():
        members[group].append(relpath)
    return members
