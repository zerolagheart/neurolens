# Training and evaluation

All commands run from the repository root after `pip install -e ".[all]"` (or use the
`scripts/*.py` wrappers, which work without installing).

## Train

```bash
neurolens train --config configs/resnet18.yaml
neurolens train --config configs/efficientnet_b0.yaml
neurolens train --config configs/custom_cnn.yaml
```

Useful overrides (any config key can be set with `--set section.key=value`):

```bash
neurolens train --config configs/resnet18.yaml --epochs 5 --batch-size 16 --device cpu
neurolens train --config configs/resnet18.yaml --set training.lr=1e-4 --set augmentation.enabled=false
neurolens train --config configs/resnet18.yaml --resume          # continue an interrupted run
```

Each run writes to `artifacts/runs/<experiment_name>/`:

| File | Content |
|---|---|
| `best.pt` | weights + metadata (architecture, classes, preprocessing, config, versions) |
| `best_meta.json` | the same metadata, readable without loading weights |
| `last.pt` | resumable state (optimiser, scheduler, scaler, early stopping, RNG) |
| `history.csv`, `history.jsonl` | per-epoch losses, accuracy, balanced accuracy, macro-F1, LR, time |
| `config.yaml`, `manifest.csv`, `manifest_meta.json` | exact configuration and split used |
| `training_summary.json` | device, epochs completed, best epoch and value |

A transfer-learning run downloads ImageNet weights from torchvision once (into the torch cache).
Set `model.pretrained: false` to avoid any download.

## Calibrate (validation split)

```bash
neurolens calibrate --run resnet18_baseline
```

Fits temperature scaling on the validation split and writes `calibration.json` (temperature,
ECE/NLL before and after, reliability bins, coverage/risk). The test set is never used.

## Evaluate (test split)

```bash
neurolens evaluate --run resnet18_baseline            # --split val is also available
neurolens compare                                      # artifacts/reports/model_comparison.{csv,json}
```

`eval/test/metrics.json` contains accuracy, balanced accuracy, macro and per-class
precision/recall/F1, the confusion matrix, the classification report, one-vs-rest ROC-AUC
(macro over classes with both positives and negatives), log loss, raw and calibrated ECE
(15 equal-width bins, top-label), parameter count, checkpoint size and latency. Plots
(confusion matrices, per-class bars, loss/accuracy curves, reliability diagram, coverage/risk)
are saved next to it, and `predictions.csv` holds per-image probabilities with relative paths.

## Benchmark latency

```bash
neurolens benchmark --run resnet18_baseline --device cpu --batch-sizes 1 32
neurolens benchmark --run resnet18_baseline --device cuda
```

Procedure: eval mode, `torch.inference_mode`, float32, preprocessing excluded, 10 warm-up
iterations discarded, 50 timed iterations with `perf_counter`, `torch.cuda.synchronize()` around
every timed iteration on CUDA, throughput = batch size / mean latency. Results record device,
thread count and versions. Only compare rows measured on the same device and batch size.

## Explain one image from the CLI

```bash
neurolens explain --run resnet18_baseline --image path/to/scan.jpg --out artifacts/explanations
```

Writes the original (at model resolution), heatmap, overlay and a JSON prediction report.
