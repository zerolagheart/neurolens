"""End-to-end smoke test on a synthetic dataset: train -> calibrate -> evaluate -> compare."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from neurolens.training.config import load_config
from neurolens.training.trainer import EarlyStopping, train
from neurolens.utils import read_json


def test_early_stopping_modes() -> None:
    stopper = EarlyStopping(patience=2, mode="max")
    assert stopper.step(0.5) and not stopper.step(0.4) and not stopper.should_stop
    assert not stopper.step(0.5) and stopper.should_stop
    minimizer = EarlyStopping(patience=0, mode="min")
    assert minimizer.step(1.0) and minimizer.step(0.5) and not minimizer.should_stop


def test_config_overrides_and_validation(tmp_path: Path) -> None:
    cfg = load_config(
        "configs/resnet18.yaml",
        ["training.epochs=2", "model.architecture=custom_cnn", "model.pretrained=false"],
    )
    assert cfg.training.epochs == 2 and cfg.model.architecture == "custom_cnn"
    with pytest.raises(ValueError, match="Unknown"):
        load_config(None, ["training.epochz=3"])
    with pytest.raises(ValueError):
        load_config(None, ["model.architecture=custom_cnn", "model.pretrained=true"])


@pytest.mark.slow
def test_train_resume_evaluate_pipeline(synthetic_dataset: Path, tmp_path: Path) -> None:
    from neurolens.evaluation.compare import build_comparison
    from neurolens.evaluation.evaluate import calibrate_run, evaluate_run
    from neurolens.inference.predictor import Predictor
    from tests.conftest import make_image

    overrides = [
        f"data.root={synthetic_dataset.as_posix()}",
        f"data.splits_dir={(tmp_path / 'splits').as_posix()}",
        f"data.audit_dir={(tmp_path / 'audit').as_posix()}",
        f"runs_dir={(tmp_path / 'runs').as_posix()}",
        "data.num_workers=0",
        "data.val_fraction=0.25",
        "experiment_name=smoke",
        "model.architecture=custom_cnn",
        "model.pretrained=false",
        "preprocessing.image_size=32",
        "training.epochs=1",
        "training.batch_size=8",
        "training.device=cpu",
        "training.early_stopping_patience=0",
    ]
    config = load_config(None, overrides)
    best = train(config)
    run_dir = best.parent
    assert (
        best.exists() and (run_dir / "last.pt").exists() and (run_dir / "best_meta.json").exists()
    )
    history = pd.read_csv(run_dir / "history.csv")
    assert list(history["epoch"]) == [1]

    with pytest.raises(FileExistsError):
        train(config)
    resumed = load_config(None, [*overrides, "training.epochs=2"])
    train(resumed, resume=True)
    assert list(pd.read_csv(run_dir / "history.csv")["epoch"]) == [1, 2]

    calibration = calibrate_run(run_dir, device="cpu", batch_size=8, num_workers=0)
    assert calibration["fitted_on"] == "val" and calibration["temperature"] > 0
    out = evaluate_run(
        run_dir, split="test", device="cpu", batch_size=8, num_workers=0, benchmark=True
    )
    metrics = read_json(out / "metrics.json")
    assert metrics["metrics"]["n_samples"] == 16
    assert metrics["calibration"]["temperature"] == pytest.approx(calibration["temperature"])
    assert metrics["latency"]["preprocessing_included"] is False
    for name in (
        "confusion_matrix.png",
        "per_class.png",
        "reliability.png",
        "curve_loss.png",
        "predictions.csv",
    ):
        assert (out / name).exists()

    table = build_comparison(tmp_path / "runs", tmp_path / "reports")
    assert set(table["status"]) == {"evaluated", "not trained"}

    predictor = Predictor.from_run_dir(run_dir, device="cpu")
    result, explanation = predictor.explain(make_image(1, 50).convert("RGB"))
    assert explanation.overlay.shape == (32, 32, 3)
    assert result.calibrated_probabilities is not None
