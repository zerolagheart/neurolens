# Generated reports

Reports are produced by the CLI into `artifacts/` (ignored by Git). Nothing in this folder is
hand-written results.

| Command | Output | Contents |
|---|---|---|
| `neurolens audit` | `artifacts/audit/audit_report.json` | counts, sizes, formats, unreadable files, duplicate and leakage summary, limitations |
| | `artifacts/audit/audit_files.csv` | per-file SHA-256, pHash, size, format, duplicate group |
| | `artifacts/audit/duplicate_pairs.csv` | exact / near-duplicate pairs with partitions and labels |
| `neurolens make-split` | `artifacts/splits/<benchmark>_seed<s>_val<v>/manifest.csv` + `manifest_meta.json` | split assignment, counts, exclusions |
| `neurolens train` | `artifacts/runs/<run>/history.csv`, `training_summary.json` | per-epoch curves, best epoch |
| `neurolens calibrate` | `artifacts/runs/<run>/calibration.json` | temperature, ECE/NLL before and after (validation) |
| `neurolens evaluate` | `artifacts/runs/<run>/eval/<split>/metrics.json` | all metrics, calibration, latency |
| | `…/predictions.csv`, `*.png` | per-image predictions, confusion matrices, per-class bars, curves, reliability, coverage-risk |
| `neurolens benchmark` | `artifacts/runs/<run>/benchmarks/pytorch_<device>.json` | latency/throughput with conditions |
| `neurolens export-onnx` | `artifacts/runs/<run>/onnx/onnx_report.json` | parity, ONNX/int8 size and latency |
| `neurolens compare` | `artifacts/reports/model_comparison.{csv,json}` | one row per run plus `not trained` architectures |

When publishing results, copy the relevant files (or a summary) into your write-up and state the
benchmark (`original` / `leakage_audited`), split seed, device and date of measurement.
