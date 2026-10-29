from __future__ import annotations

import numpy as np
import pytest

from neurolens.evaluation.calibration import (
    AbstentionPolicy,
    coverage_risk,
    entropy,
    expected_calibration_error,
    fit_temperature,
    normalized_entropy,
    reliability_bins,
)
from neurolens.evaluation.metrics import compute_classification_metrics, softmax

CLASSES = ["a", "b", "c"]


def test_softmax_normalizes_and_temperature_preserves_argmax() -> None:
    logits = np.array([[2.0, 1.0, -1.0], [0.0, 0.0, 5.0]])
    for t in (0.5, 1.0, 3.0):
        p = softmax(logits, t)
        assert np.allclose(p.sum(1), 1.0)
        assert (p.argmax(1) == logits.argmax(1)).all()
    assert softmax(logits, 3.0).max() < softmax(logits, 1.0).max()


def test_metrics_on_known_predictions() -> None:
    y = np.array([0, 0, 1, 1, 2, 2])
    probs = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.2, 0.7, 0.1],  # second sample misclassified as b
            [0.1, 0.8, 0.1],
            [0.1, 0.6, 0.3],
            [0.1, 0.1, 0.8],
            [0.2, 0.2, 0.6],
        ]
    )
    m = compute_classification_metrics(y, probs, CLASSES)
    assert m["accuracy"] == pytest.approx(5 / 6)
    assert m["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [0, 0, 2]]
    assert m["per_class"]["a"]["recall"] == pytest.approx(0.5)
    assert m["per_class"]["b"]["precision"] == pytest.approx(2 / 3)
    assert m["balanced_accuracy"] == pytest.approx((0.5 + 1 + 1) / 3)
    assert m["macro_f1"] == pytest.approx(np.mean([2 / 3, 0.8, 1.0]))
    assert 0.5 < m["roc_auc"]["macro"] <= 1.0
    assert "precision" in m["classification_report"]


def test_metrics_shape_validation_and_missing_class_auc() -> None:
    with pytest.raises(ValueError):
        compute_classification_metrics([0, 1], np.ones((2, 2)) / 2, CLASSES)
    y = np.array([0, 0, 1, 1])
    probs = np.array([[0.8, 0.1, 0.1], [0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.3, 0.6, 0.1]])
    m = compute_classification_metrics(y, probs, CLASSES)
    assert m["roc_auc"]["per_class"][2] is None
    assert m["roc_auc"]["macro"] == pytest.approx(1.0)


def test_ece_perfect_and_overconfident() -> None:
    labels = np.array([0, 1, 0, 1])
    perfect = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(perfect, labels) == pytest.approx(0.0)
    overconfident = np.array([[0.95, 0.05]] * 4)  # always predicts class 0, 50% accuracy
    assert expected_calibration_error(overconfident, labels, n_bins=10) == pytest.approx(0.45)
    bins = reliability_bins(overconfident, labels, n_bins=10)
    assert sum(b["count"] for b in bins) == 4 and bins[9]["count"] == 4


def test_temperature_scaling_reduces_overconfidence() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, 600)
    clean = np.eye(3)[labels] * 2.0 + rng.normal(0, 1.0, (600, 3))
    logits = clean * 4.0  # sharpen -> overconfident
    t = fit_temperature(logits, labels)
    assert t > 1.5
    before = expected_calibration_error(softmax(logits), labels)
    after = expected_calibration_error(softmax(logits, t), labels)
    assert after < before


def test_entropy_bounds_and_abstention() -> None:
    uniform = np.full(4, 0.25)
    onehot = np.array([1.0, 0, 0, 0])
    assert normalized_entropy(uniform[None])[0] == pytest.approx(1.0)
    assert normalized_entropy(onehot[None])[0] == pytest.approx(0.0, abs=1e-9)
    assert entropy(uniform) == pytest.approx(np.log(4))

    policy = AbstentionPolicy(min_confidence=0.7, max_normalized_entropy=0.6)
    abstain, reasons = policy.evaluate(uniform)
    assert abstain and len(reasons) == 2
    assert policy.evaluate(np.array([0.97, 0.01, 0.01, 0.01])) == (False, [])
    assert AbstentionPolicy(enabled=False).evaluate(uniform) == (False, [])


def test_coverage_risk_monotone_coverage() -> None:
    probs = softmax(np.random.default_rng(1).normal(0, 2, (200, 4)))
    labels = np.random.default_rng(2).integers(0, 4, 200)
    rows = coverage_risk(probs, labels)
    coverages = [r["coverage"] for r in rows]
    assert coverages == sorted(coverages, reverse=True)
    assert rows[0]["coverage"] == pytest.approx(1.0)
