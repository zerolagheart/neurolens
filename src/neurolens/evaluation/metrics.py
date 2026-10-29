"""Classification metrics computed from labels and predicted probabilities.

ROC-AUC strategy: one-vs-rest (OvR). For each class ``k`` the scores ``p[:, k]`` are ranked
against the binary target ``y == k``; the macro ROC-AUC is the unweighted mean over classes that
have both positive and negative examples. Classes without positives are reported as ``null``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def one_vs_rest_auc(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> dict[str, Any]:
    per_class: list[float | None] = []
    for k in range(num_classes):
        target = (y_true == k).astype(int)
        if target.min() == target.max():
            per_class.append(None)
        else:
            per_class.append(float(roc_auc_score(target, probs[:, k])))
    valid = [v for v in per_class if v is not None]
    return {
        "strategy": "one-vs-rest, macro average over classes with positives and negatives",
        "macro": float(np.mean(valid)) if valid else None,
        "per_class": per_class,
    }


def compute_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    probs: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, Any]:
    """All headline metrics for a set of predictions. ``probs`` has shape ``(N, K)``."""
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum(axis=1, keepdims=True)  # remove float32 rounding drift
    num_classes = len(class_names)
    if probs.ndim != 2 or probs.shape != (len(y_true), num_classes):
        raise ValueError(f"probs must have shape ({len(y_true)}, {num_classes}), got {probs.shape}")
    labels = list(range(num_classes))
    y_pred = probs.argmax(axis=1)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    auc = one_vs_rest_auc(y_true, probs, num_classes)
    per_class = {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "roc_auc_ovr": auc["per_class"][i],
        }
        for i, name in enumerate(class_names)
    }
    return {
        "n_samples": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "log_loss": float(log_loss(y_true, np.clip(probs, 1e-12, 1.0), labels=labels)),
        "roc_auc": auc,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=list(class_names), zero_division=0, digits=4
        ),
    }
