"""Calibration and uncertainty: temperature scaling, ECE, entropy, abstention, coverage/risk.

* Raw probabilities are ``softmax(logits)``. Calibrated probabilities are
  ``softmax(logits / T)`` with a single temperature ``T`` fitted by minimising the negative
  log-likelihood on the **validation** split. Temperature scaling never changes the argmax.
* ECE uses the top-label confidence and ``n_bins`` equal-width bins over ``[0, 1]``
  (first bin closed on both sides, the rest ``(lo, hi]``):
  ``ECE = sum_b |B_b|/N * |acc(B_b) - conf(B_b)|``.
* The abstention thresholds are configuration, not clinically validated safety thresholds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from neurolens.evaluation.metrics import softmax


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    """Fit ``T > 0`` minimising NLL of ``softmax(logits / T)`` (optimises ``log T`` with LBFGS)."""
    logits_t = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
    labels_t = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits_t / log_t.exp(), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_t.exp().item())
    if not np.isfinite(temperature) or temperature <= 0:
        raise RuntimeError("Temperature scaling diverged")
    return temperature


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    picked = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.log(picked).mean())


def reliability_bins(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> list[dict[str, Any]]:
    confidences = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # bin index b means confidence in (edges[b], edges[b+1]]; confidence 0 goes to bin 0
    idx = np.clip(np.searchsorted(edges, confidences, side="left") - 1, 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        bins.append(
            {
                "lower": float(edges[b]),
                "upper": float(edges[b + 1]),
                "count": count,
                "accuracy": float(correct[mask].mean()) if count else None,
                "confidence": float(confidences[mask].mean()) if count else None,
            }
        )
    return bins


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bins = reliability_bins(probs, labels, n_bins)
    total = len(labels)
    return float(
        sum(b["count"] / total * abs(b["accuracy"] - b["confidence"]) for b in bins if b["count"])
    )


def entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy in nats for each row of ``probs``."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=-1)


def normalized_entropy(probs: np.ndarray) -> np.ndarray:
    """Entropy divided by ``log(K)``: 0 = one-hot, 1 = uniform."""
    return entropy(probs) / np.log(np.asarray(probs).shape[-1])


@dataclass
class AbstentionPolicy:
    """Accept a prediction only if confidence and entropy criteria hold.

    ``min_confidence`` applies to the calibrated top-class probability when a temperature is
    available, otherwise to the raw softmax score.
    """

    enabled: bool = True
    min_confidence: float = 0.70
    max_normalized_entropy: float | None = 0.60

    def evaluate(self, probs: np.ndarray) -> tuple[bool, list[str]]:
        """Return ``(abstain, reasons)`` for a single probability vector."""
        if not self.enabled:
            return False, []
        reasons: list[str] = []
        confidence = float(np.max(probs))
        if confidence < self.min_confidence:
            reasons.append(
                f"top-class probability {confidence:.2f} is below the configured threshold "
                f"{self.min_confidence:.2f}"
            )
        if self.max_normalized_entropy is not None:
            h = float(normalized_entropy(probs[None, :])[0])
            if h > self.max_normalized_entropy:
                reasons.append(
                    f"normalised entropy {h:.2f} exceeds the configured limit "
                    f"{self.max_normalized_entropy:.2f}"
                )
        return bool(reasons), reasons

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def coverage_risk(
    probs: np.ndarray, labels: np.ndarray, thresholds: np.ndarray | None = None
) -> list[dict[str, float]]:
    """Coverage (fraction accepted) and risk (error rate among accepted) per threshold."""
    thresholds = np.linspace(0.25, 0.99, 38) if thresholds is None else thresholds
    confidences = probs.max(axis=1)
    errors = probs.argmax(axis=1) != labels
    rows = []
    for threshold in thresholds:
        accepted = confidences >= threshold
        n = int(accepted.sum())
        rows.append(
            {
                "threshold": round(float(threshold), 4),
                "coverage": n / len(labels),
                "risk": float(errors[accepted].mean()) if n else 0.0,
                "accepted": n,
            }
        )
    return rows


def calibration_summary(
    logits: np.ndarray, labels: np.ndarray, temperature: float | None, n_bins: int = 15
) -> dict[str, Any]:
    raw = softmax(logits)
    summary: dict[str, Any] = {
        "ece_bins": n_bins,
        "binning": "equal-width, top-label confidence",
        "raw": {
            "ece": expected_calibration_error(raw, labels, n_bins),
            "nll": nll(raw, labels),
            "reliability": reliability_bins(raw, labels, n_bins),
            "coverage_risk": coverage_risk(raw, labels),
        },
    }
    if temperature is not None:
        cal = softmax(logits, temperature)
        summary["temperature"] = temperature
        summary["calibrated"] = {
            "ece": expected_calibration_error(cal, labels, n_bins),
            "nll": nll(cal, labels),
            "reliability": reliability_bins(cal, labels, n_bins),
            "coverage_risk": coverage_risk(cal, labels),
        }
    return summary
