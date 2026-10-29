from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from neurolens.data.audit import (
    AuditConfig,
    hamming_distance,
    near_duplicate_pairs,
    perceptual_hash,
    run_audit,
    save_audit,
)
from neurolens.data.discovery import discover_dataset
from neurolens.data.splits import build_manifest
from tests.conftest import make_image


def test_perceptual_hash_robust_to_reencoding_and_resizing() -> None:
    image = make_image(5, size=128)
    resized = image.resize((96, 96))
    other = make_image(77, size=128)
    h1, h2, h3 = perceptual_hash(image), perceptual_hash(resized), perceptual_hash(other)
    assert hamming_distance(h1, h2) <= 4
    assert hamming_distance(h1, h3) > hamming_distance(h1, h2)


def test_near_duplicate_pairs_matches_bruteforce() -> None:
    rng = np.random.default_rng(0)
    hashes = rng.integers(0, 2**63, 50, dtype=np.uint64)
    hashes[10] = hashes[3] ^ np.uint64(0b101)  # distance 2
    pairs = near_duplicate_pairs(hashes, threshold=2, chunk=7)
    brute = [
        (i, j, bin(int(hashes[i]) ^ int(hashes[j])).count("1"))
        for i in range(50)
        for j in range(i + 1, 50)
        if bin(int(hashes[i]) ^ int(hashes[j])).count("1") <= 2
    ]
    assert sorted(pairs) == sorted(brute)
    assert (3, 10, 2) in pairs


def test_audit_finds_cross_split_duplicates(synthetic_dataset: Path, tmp_path: Path) -> None:
    train_file = next((synthetic_dataset / "Training" / "glioma").glob("*.png"))
    # Exact copy into the test split, and a resized (near) copy into another test class.
    shutil.copy(train_file, synthetic_dataset / "Testing" / "glioma" / "copy.png")
    from PIL import Image

    Image.open(train_file).resize((40, 40)).save(
        synthetic_dataset / "Testing" / "meningioma" / "near.png"
    )
    (synthetic_dataset / "Testing" / "pituitary" / "broken.png").write_bytes(b"garbage")

    layout = discover_dataset(synthetic_dataset)
    result = run_audit(layout, AuditConfig(near_duplicate_threshold=4, workers=2))
    report = result.report
    assert report["exact_duplicates"]["cross_partition_pairs"] >= 1
    assert report["leakage"]["test_files_with_match_in_train"] >= 1
    assert report["near_duplicates"]["label_conflict_pairs"] >= 1
    assert [u["relpath"] for u in report["unreadable_files"]] == ["Testing/pituitary/broken.png"]
    assert report["limitations"] and "patient" in report["limitations"][0]

    paths = save_audit(result, tmp_path / "audit")
    assert all(p.exists() for p in paths.values())

    # The leakage-audited manifest drops the duplicated test images but keeps the original files.
    groups = dict(zip(result.files["relpath"], result.files["dup_group"], strict=True))
    frame, meta = build_manifest(
        layout,
        val_fraction=0.25,
        seed=0,
        benchmark="leakage_audited",
        dup_groups=groups,
        unreadable=[u["relpath"] for u in report["unreadable_files"]],
    )
    test_paths = set(frame.loc[frame["split"] == "test", "relpath"])
    assert "Testing/glioma/copy.png" not in test_paths
    assert meta["benchmark"] == "leakage_audited"
    assert (synthetic_dataset / "Testing" / "glioma" / "copy.png").exists()
    # train/val never share a duplicate group
    tr = set(frame.loc[frame["split"] == "train", "group_id"])
    va = set(frame.loc[frame["split"] == "val", "group_id"])
    assert tr.isdisjoint(va)
