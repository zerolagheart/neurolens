"""Aggregate evaluated runs into a model comparison table.

Architectures without a trained run are listed with status ``not trained`` and empty metrics
instead of being omitted, so missing experiments stay visible. Latency columns carry their
device; rows measured on different devices must not be compared directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from neurolens.models.factory import ARCHITECTURES
from neurolens.utils import read_json, utc_now, write_json


def _latency(bench: dict[str, Any] | None, batch_size: int) -> dict[str, Any] | None:
    if not bench:
        return None
    return next((r for r in bench.get("latency", []) if r["batch_size"] == batch_size), None)


def collect_runs(runs_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for run_dir in sorted(p for p in Path(runs_dir).glob("*") if (p / "best.pt").is_file()):
        meta_path = run_dir / "best_meta.json"
        meta = read_json(meta_path) if meta_path.exists() else {}
        metrics_path = run_dir / "eval" / "test" / "metrics.json"
        evaluation = read_json(metrics_path) if metrics_path.exists() else None
        bench_files = sorted((run_dir / "benchmarks").glob("pytorch_*.json"))
        bench = read_json(bench_files[0]) if bench_files else None
        onnx_path = run_dir / "onnx" / "onnx_report.json"
        onnx = read_json(onnx_path) if onnx_path.exists() else None

        row: dict[str, Any] = {
            "run": run_dir.name,
            "architecture": meta.get(
                "architecture", evaluation["architecture"] if evaluation else None
            ),
            "benchmark": meta.get("benchmark"),
            "status": "evaluated" if evaluation else "trained, not evaluated",
            "best_epoch": meta.get("epoch"),
        }
        if evaluation:
            m = evaluation["metrics"]
            cal = evaluation["calibration"]
            row.update(
                test_n=m["n_samples"],
                accuracy=m["accuracy"],
                balanced_accuracy=m["balanced_accuracy"],
                macro_precision=m["macro_precision"],
                macro_recall=m["macro_recall"],
                macro_f1=m["macro_f1"],
                roc_auc_ovr_macro=m["roc_auc"]["macro"],
                ece_raw=cal["ece_raw"],
                ece_calibrated=cal["ece_calibrated"],
                temperature=cal["temperature"],
                parameters=evaluation["model"]["parameters"],
                checkpoint_mb=evaluation["model"]["checkpoint_size_mb"],
            )
        lat1, lat32 = _latency(bench, 1), _latency(bench, 32)
        if bench:
            row.update(
                latency_device=bench["device"],
                latency_b1_ms=lat1["mean_ms"] if lat1 else None,
                throughput_b32_img_s=lat32["throughput_img_s"] if lat32 else None,
            )
        if onnx:
            row.update(
                onnx_mb=onnx["onnx"]["size_mb"],
                onnx_parity_ok=onnx["parity"]["within_tolerance"],
                onnx_b1_ms=_onnx_latency(onnx, "onnx", 1),
            )
        rows.append(row)
    trained = {r["architecture"] for r in rows}
    for name in ARCHITECTURES:
        if name not in trained:
            rows.append({"run": None, "architecture": name, "status": "not trained"})
    return rows


def _onnx_latency(report: dict[str, Any], key: str, batch_size: int) -> float | None:
    for row in report.get("benchmark", {}).get(key, []):
        if row["batch_size"] == batch_size:
            return row["mean_ms"]
    return None


def build_comparison(runs_dir: Path, out_dir: Path) -> pd.DataFrame:
    rows = collect_runs(runs_dir)
    table = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "model_comparison.csv", index=False)
    write_json(
        out_dir / "model_comparison.json",
        {
            "generated_at": utc_now(),
            "notes": [
                "Metrics are offline evaluations on each run's test split.",
                "Runs on the leakage_audited benchmark use a different, smaller test set and are "
                "not comparable with runs on the original benchmark.",
                "Latency values are only comparable between rows with the same latency_device.",
            ],
            "rows": rows,
        },
    )
    return table
