"""Evaluate a trained run on its validation or test split, and fit temperature scaling.

Outputs (per run)::

    <run>/calibration.json             temperature fitted on the validation split
    <run>/eval/<split>/metrics.json    machine-readable metrics (raw and calibrated)
    <run>/eval/<split>/predictions.csv per-image predictions (relative paths only)
    <run>/eval/<split>/*.png           confusion matrices, per-class bars, curves, reliability
    <run>/benchmarks/pytorch_<device>.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from neurolens.data.splits import load_manifest, split_items
from neurolens.data.transforms import build_eval_transform
from neurolens.evaluation import plots
from neurolens.evaluation.benchmark import benchmark_model
from neurolens.evaluation.calibration import calibration_summary, fit_temperature
from neurolens.evaluation.metrics import compute_classification_metrics, softmax
from neurolens.training.checkpoint import model_from_checkpoint, preprocess_from_metadata
from neurolens.training.trainer import make_loader
from neurolens.utils import file_size_mb, get_logger, read_json, resolve_device, utc_now, write_json

log = get_logger("evaluate")


def collect_logits(
    model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    with torch.inference_mode():
        for images, targets in loader:
            logits.append(model(images.to(device)).float().cpu())
            labels.append(targets)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def _prepare(
    run_dir: Path, split: str, data_root: str | None, device: str, batch_size: int, num_workers: int
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    torch_device = resolve_device(device)
    model, metadata = model_from_checkpoint(run_dir / "best.pt", torch_device)
    root = Path(data_root or metadata["experiment"]["data"]["root"])
    frame, _ = load_manifest(run_dir / "manifest.csv", root)
    items = split_items(frame, split)
    if not items:
        raise ValueError(f"Split '{split}' is empty in {run_dir.name}/manifest.csv")
    preprocess = preprocess_from_metadata(metadata)
    loader = make_loader(
        items, build_eval_transform(preprocess), batch_size, False, num_workers, 0, torch_device
    )
    logits, labels = collect_logits(model, loader, torch_device)
    return {
        "model": model,
        "metadata": metadata,
        "device": torch_device,
        "logits": logits,
        "labels": labels,
        "frame": frame[frame["split"] == split].reset_index(drop=True),
        "preprocess": preprocess,
        "items": items,
    }


def load_temperature(run_dir: Path) -> float | None:
    path = Path(run_dir) / "calibration.json"
    return float(read_json(path)["temperature"]) if path.exists() else None


def calibrate_run(
    run_dir: Path,
    data_root: str | None = None,
    device: str = "auto",
    batch_size: int = 64,
    num_workers: int = 2,
    n_bins: int = 15,
) -> dict[str, Any]:
    """Fit temperature scaling on the validation split (never on test) and save it."""
    ctx = _prepare(run_dir, "val", data_root, device, batch_size, num_workers)
    temperature = fit_temperature(ctx["logits"], ctx["labels"])
    summary = calibration_summary(ctx["logits"], ctx["labels"], temperature, n_bins)
    result = {
        "method": "temperature_scaling",
        "temperature": temperature,
        "fitted_on": "val",
        "n_samples": len(ctx["labels"]),
        "fitted_at": utc_now(),
        "model_version": ctx["metadata"].get("model_version"),
        "ece_bins": n_bins,
        "ece_before": summary["raw"]["ece"],
        "ece_after": summary["calibrated"]["ece"],
        "nll_before": summary["raw"]["nll"],
        "nll_after": summary["calibrated"]["nll"],
        "summary": summary,
    }
    write_json(Path(run_dir) / "calibration.json", result)
    return result


def evaluate_run(
    run_dir: Path,
    split: str = "test",
    data_root: str | None = None,
    device: str = "auto",
    batch_size: int = 64,
    num_workers: int = 2,
    benchmark: bool = True,
    n_bins: int = 15,
) -> Path:
    run_dir = Path(run_dir)
    ctx = _prepare(run_dir, split, data_root, device, batch_size, num_workers)
    metadata, logits, labels = ctx["metadata"], ctx["logits"], ctx["labels"]
    class_names = list(metadata["class_names"])
    out_dir = run_dir / "eval" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_probs = softmax(logits)
    temperature = load_temperature(run_dir)
    if temperature is None:
        log.warning(
            "No calibration.json for %s; run `neurolens calibrate` first for calibrated metrics",
            run_dir.name,
        )
    calibration = calibration_summary(logits, labels, temperature, n_bins)

    metrics_raw = compute_classification_metrics(labels, raw_probs, class_names)
    result: dict[str, Any] = {
        "run": run_dir.name,
        "architecture": metadata["architecture"],
        "model_version": metadata.get("model_version"),
        "checkpoint_epoch": metadata.get("epoch"),
        "benchmark": metadata.get("benchmark"),
        "split": split,
        "evaluated_at": utc_now(),
        "evaluation_type": "offline",
        "class_names": class_names,
        "metrics": metrics_raw,
        "calibration": {
            "temperature": temperature,
            "ece_bins": n_bins,
            "ece_raw": calibration["raw"]["ece"],
            "ece_calibrated": calibration.get("calibrated", {}).get("ece"),
            "nll_raw": calibration["raw"]["nll"],
            "nll_calibrated": calibration.get("calibrated", {}).get("nll"),
            "details": calibration,
        },
        "model": {
            "parameters": sum(p.numel() for p in ctx["model"].parameters()),
            "checkpoint_size_mb": file_size_mb(run_dir / "best.pt"),
        },
    }
    if split == "test" and metadata.get("benchmark") == "leakage_audited":
        result["note"] = "Leakage-audited test subset: not comparable with the original benchmark."

    # Per-image predictions with relative paths only.
    pred_frame = pd.DataFrame(
        {
            "relpath": ctx["frame"]["relpath"],
            "label": [class_names[i] for i in labels],
            "predicted": [class_names[i] for i in raw_probs.argmax(1)],
            "raw_confidence": raw_probs.max(1),
        }
    )
    for k, name in enumerate(class_names):
        pred_frame[f"p_{name}"] = raw_probs[:, k]
    if temperature is not None:
        pred_frame["calibrated_confidence"] = softmax(logits, temperature).max(1)
    pred_frame.to_csv(out_dir / "predictions.csv", index=False)

    plots.plot_confusion_matrix(
        metrics_raw["confusion_matrix"], class_names, out_dir / "confusion_matrix.png"
    )
    plots.plot_confusion_matrix(
        metrics_raw["confusion_matrix"],
        class_names,
        out_dir / "confusion_matrix_normalized.png",
        normalize=True,
    )
    plots.plot_per_class(metrics_raw["per_class"], out_dir / "per_class.png")
    plots.plot_reliability(calibration, out_dir / "reliability.png")
    plots.plot_coverage_risk(calibration, out_dir / "coverage_risk.png")
    if (run_dir / "history.csv").exists():
        plots.plot_training_curves(pd.read_csv(run_dir / "history.csv"), out_dir)

    if benchmark:
        bench = benchmark_model(
            ctx["model"],
            ctx["device"],
            ctx["preprocess"],
            [1, 32],
            sample_image=Path(ctx["items"][0][0]),
            checkpoint=run_dir / "best.pt",
        )
        write_json(run_dir / "benchmarks" / f"pytorch_{ctx['device'].type}.json", bench)
        result["latency"] = bench

    write_json(out_dir / "metrics.json", result)
    log.info(
        "%s %s: accuracy=%.4f balanced=%.4f macro_f1=%.4f",
        run_dir.name,
        split,
        metrics_raw["accuracy"],
        metrics_raw["balanced_accuracy"],
        metrics_raw["macro_f1"],
    )
    return out_dir
