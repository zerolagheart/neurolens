"""Static Matplotlib figures for offline reports (the Streamlit app renders Plotly versions)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neurolens.data.classes import display_name

ACCENT = "#0e9f9a"
SECOND = "#334155"


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion_matrix(
    matrix: Sequence[Sequence[int]], class_names: Sequence[str], path: Path, normalize: bool = False
) -> Path:
    cm = np.asarray(matrix, dtype=float)
    if normalize:
        cm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    labels = [display_name(c) for c in class_names]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalised)" if normalize else "Confusion matrix (counts)")
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=9,
            )
    return _save(fig, path)


def plot_per_class(per_class: dict[str, dict[str, Any]], path: Path) -> Path:
    names = list(per_class)
    metrics = ["precision", "recall", "f1"]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for k, metric in enumerate(metrics):
        ax.bar(x + (k - 1) * 0.26, [per_class[n][metric] for n in names], 0.26, label=metric)
    ax.set_xticks(x, [display_name(n) for n in names])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Per-class metrics")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)


def plot_training_curves(history: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    for metric, title in (("loss", "Loss"), ("accuracy", "Accuracy")):
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        ax.plot(
            history["epoch"], history[f"train_{metric}"], marker="o", color=SECOND, label="train"
        )
        ax.plot(
            history["epoch"], history[f"val_{metric}"], marker="o", color=ACCENT, label="validation"
        )
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.set_title(f"{title} curves")
        ax.grid(alpha=0.3)
        ax.legend()
        paths.append(_save(fig, Path(out_dir) / f"curve_{metric}.png"))
    return paths


def plot_reliability(summary: dict[str, Any], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    for key, color in (("raw", SECOND), ("calibrated", ACCENT)):
        if key not in summary:
            continue
        bins = [b for b in summary[key]["reliability"] if b["count"]]
        ax.plot(
            [b["confidence"] for b in bins],
            [b["accuracy"] for b in bins],
            marker="o",
            color=color,
            label=f"{key} (ECE {summary[key]['ece']:.3f})",
        )
    ax.set_xlabel("mean confidence in bin")
    ax.set_ylabel("accuracy in bin")
    ax.set_title(f"Reliability diagram ({summary['ece_bins']} bins)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, path)


def plot_coverage_risk(summary: dict[str, Any], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for key, color in (("raw", SECOND), ("calibrated", ACCENT)):
        if key not in summary:
            continue
        rows = summary[key]["coverage_risk"]
        ax.plot(
            [r["coverage"] for r in rows],
            [r["risk"] for r in rows],
            marker=".",
            color=color,
            label=key,
        )
    ax.set_xlabel("coverage (fraction of images accepted)")
    ax.set_ylabel("error rate on accepted images")
    ax.set_title("Coverage vs. risk under confidence thresholds")
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, path)
