from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from neurolens.data.classes import CANONICAL_CLASSES, normalize_class_name
from neurolens.data.dataset import CorruptImageError, MRIDataset, check_image, filter_readable
from neurolens.data.discovery import DatasetStructureError, discover_dataset
from neurolens.data.splits import build_manifest, load_manifest, save_manifest, stratified_split
from neurolens.data.transforms import (
    AugmentationConfig,
    PreprocessConfig,
    build_eval_transform,
    build_train_transform,
    load_image_rgb,
    preprocess_image,
)


def test_class_aliases_and_order() -> None:
    assert tuple(sorted(CANONICAL_CLASSES)) == CANONICAL_CLASSES
    assert normalize_class_name("No_Tumor") == "notumor"
    assert normalize_class_name("pituitary_tumor") == "pituitary"
    assert normalize_class_name("cat") is None


def test_discovery_finds_splits_and_labels(synthetic_dataset: Path) -> None:
    layout = discover_dataset(synthetic_dataset)
    assert layout.class_names == CANONICAL_CLASSES
    assert len(layout.by_split("train")) == 48
    assert len(layout.by_split("test")) == 16
    for sample in layout.samples:
        assert CANONICAL_CLASSES[sample.label] == sample.class_name
        assert sample.path.parent.name == sample.class_name


def test_discovery_handles_wrapper_dir_and_alias_names(
    tmp_path: Path, synthetic_dataset: Path
) -> None:
    wrapped = tmp_path / "download" / "archive"
    wrapped.parent.mkdir(parents=True)
    synthetic_dataset.rename(wrapped)
    (wrapped / "Training" / "notumor").rename(wrapped / "Training" / "no_tumor")
    (wrapped / "Training" / "glioma" / "readme.txt").write_text("not an image")
    layout = discover_dataset(tmp_path / "download")
    assert len(layout.by_split("train")) == 48
    assert [p.name for p in layout.ignored_files] == ["readme.txt"]


def test_discovery_reports_missing_class(synthetic_dataset: Path) -> None:
    for f in (synthetic_dataset / "Testing" / "glioma").iterdir():
        f.unlink()
    (synthetic_dataset / "Testing" / "glioma").rmdir()
    with pytest.raises(DatasetStructureError, match="missing class folders"):
        discover_dataset(synthetic_dataset)


def test_discovery_missing_root(tmp_path: Path) -> None:
    with pytest.raises(DatasetStructureError, match="does not exist"):
        discover_dataset(tmp_path / "nope")


def test_corrupt_image_detection_and_dataset_error(tmp_path: Path) -> None:
    good = tmp_path / "good.png"
    Image.new("L", (40, 40), 128).save(good)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"\xff\xd8\xff\xe0 truncated nonsense")
    assert check_image(good) is None
    assert check_image(bad) is not None

    readable, unreadable = filter_readable([(good, 0), (bad, 1)])
    assert readable == [(good, 0)]
    assert unreadable[0][0] == bad

    dataset = MRIDataset([(bad, 1)], build_eval_transform(PreprocessConfig(image_size=32)))
    with pytest.raises(CorruptImageError):
        dataset[0]


def test_dataset_returns_normalized_tensors(synthetic_dataset: Path) -> None:
    layout = discover_dataset(synthetic_dataset)
    items = [(s.path, s.label) for s in layout.by_split("train")[:4]]
    dataset = MRIDataset(items, build_eval_transform(PreprocessConfig(image_size=32)))
    tensor, label = dataset[0]
    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == torch.float32
    assert label == items[0][1]


def test_preprocessing_grayscale_and_16bit_to_rgb(tmp_path: Path) -> None:
    import numpy as np

    path = tmp_path / "deep.png"
    Image.fromarray((np.arange(64 * 64).reshape(64, 64) * 10).astype(np.uint16)).save(path)
    image = load_image_rgb(path)
    assert image.mode == "RGB"
    cfg = PreprocessConfig(image_size=40)
    tensor = preprocess_image(image, cfg)
    assert tensor.shape == (1, 3, 40, 40)
    # Grayscale replicated to 3 channels -> channels differ only by normalisation constants.
    unnorm = (
        tensor[0] * torch.tensor(cfg.std)[:, None, None] + torch.tensor(cfg.mean)[:, None, None]
    )
    assert torch.allclose(unnorm[0], unnorm[1], atol=1e-5)


def test_eval_transform_is_deterministic_and_train_transform_augments() -> None:
    from tests.conftest import make_image

    image = make_image(3, size=64).convert("RGB")
    cfg = PreprocessConfig(image_size=32)
    eval_t = build_eval_transform(cfg)
    assert torch.equal(eval_t(image), eval_t(image))
    torch.manual_seed(0)
    train_t = build_train_transform(cfg, AugmentationConfig(rotation_degrees=30))
    outputs = [train_t(image) for _ in range(4)]
    assert outputs[0].shape == (3, 32, 32)
    assert not all(torch.equal(outputs[0], o) for o in outputs[1:])
    assert build_train_transform(cfg, AugmentationConfig(enabled=False))(image).equal(eval_t(image))


def test_preprocess_config_roundtrip() -> None:
    cfg = PreprocessConfig(image_size=128)
    assert PreprocessConfig.from_dict(cfg.to_dict()) == cfg


def test_stratified_split_is_reproducible_and_stratified() -> None:
    labels = [i % 4 for i in range(200)]
    a = stratified_split(labels, 0.2, seed=7)
    b = stratified_split(labels, 0.2, seed=7)
    c = stratified_split(labels, 0.2, seed=8)
    assert a == b
    assert a != c
    train, val = a
    assert set(train).isdisjoint(val) and len(train) + len(val) == 200
    assert [sum(labels[i] == k for i in val) for k in range(4)] == [10, 10, 10, 10]


def test_group_split_keeps_groups_together() -> None:
    labels = [i % 4 for i in range(120)]
    groups = [f"g{i // 3}" for i in range(120)]
    train, val = stratified_split(labels, 0.25, seed=1, groups=groups)
    assert {groups[i] for i in train}.isdisjoint({groups[i] for i in val})


def test_manifest_original_keeps_test_and_roundtrips(
    synthetic_dataset: Path, tmp_path: Path
) -> None:
    layout = discover_dataset(synthetic_dataset)
    frame, meta = build_manifest(layout, val_fraction=0.25, seed=3)
    assert meta["benchmark"] == "original"
    assert (frame["split"] == "test").sum() == 16
    assert (frame["split"] == "val").sum() == 12
    assert not frame["relpath"].str.contains(":").any()  # relative paths only
    path = save_manifest(frame, meta, tmp_path / "split")
    loaded, loaded_meta = load_manifest(path, synthetic_dataset)
    assert loaded_meta["seed"] == 3
    assert all(p.exists() for p in loaded["path"])


def test_manifest_excludes_unreadable_files(synthetic_dataset: Path) -> None:
    bad = synthetic_dataset / "Training" / "glioma" / "broken.jpg"
    bad.write_bytes(b"not really a jpeg")
    layout = discover_dataset(synthetic_dataset)
    frame, meta = build_manifest(layout, val_fraction=0.25, seed=0)
    assert "Training/glioma/broken.jpg" not in set(frame["relpath"])
    assert any(e["relpath"] == "Training/glioma/broken.jpg" for e in meta["excluded"])


def test_leakage_audited_manifest_requires_groups(synthetic_dataset: Path) -> None:
    layout = discover_dataset(synthetic_dataset)
    with pytest.raises(ValueError, match="audit"):
        build_manifest(layout, benchmark="leakage_audited")
