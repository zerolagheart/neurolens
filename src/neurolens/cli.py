"""``neurolens`` command-line interface.

Examples::

    neurolens audit --data-root data/raw/brain-tumor-mri-dataset
    neurolens train --config configs/resnet18.yaml
    neurolens calibrate --run resnet18_baseline
    neurolens evaluate --run resnet18_baseline --split test
    neurolens export-onnx --run resnet18_baseline --quantize
    neurolens compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neurolens.utils import configure_logging, get_logger

log = get_logger("cli")
DEFAULT_RUNS_DIR = Path("artifacts/runs")


def resolve_run(run: str, runs_dir: Path = DEFAULT_RUNS_DIR) -> Path:
    """Accept a run name (under ``runs_dir``) or a run directory path."""
    candidate = Path(run)
    run_dir = candidate if candidate.is_dir() else Path(runs_dir) / run
    if not (run_dir / "best.pt").is_file():
        raise SystemExit(
            f"No best.pt checkpoint in '{run_dir}'. Train it first, e.g.\n"
            f"  neurolens train --config configs/resnet18.yaml"
        )
    return run_dir


def _cmd_audit(args: argparse.Namespace) -> int:
    from neurolens.data.audit import AuditConfig, run_audit, save_audit
    from neurolens.data.discovery import discover_dataset

    layout = discover_dataset(args.data_root)
    partitions = None
    if args.manifest:
        import pandas as pd

        frame = pd.read_csv(args.manifest)
        partitions = dict(zip(frame["relpath"], frame["split"], strict=True))
    result = run_audit(
        layout,
        AuditConfig(near_duplicate_threshold=args.threshold, workers=args.workers),
        partitions,
    )
    paths = save_audit(result, Path(args.out))
    report = result.report
    print(f"Audited {report['total_files']} files -> {paths['report'].as_posix()}")
    print(f"  unreadable: {len(report['unreadable_files'])}")
    print(
        f"  exact duplicate pairs: {report['exact_duplicates']['pairs']} "
        f"(cross-partition {report['exact_duplicates']['cross_partition_pairs']})"
    )
    print(
        f"  near-duplicate candidate pairs (pHash <= {args.threshold}): "
        f"{report['near_duplicates']['pairs']} "
        f"(cross-partition {report['near_duplicates']['cross_partition_pairs']})"
    )
    for key, value in report["leakage"].items():
        print(f"  {key}: {value}")
    return 0


def _load_config(args: argparse.Namespace):
    from neurolens.training.config import load_config

    overrides = list(args.set or [])
    mapping = {
        "experiment_name": "experiment_name",
        "arch": "model.architecture",
        "epochs": "training.epochs",
        "batch_size": "training.batch_size",
        "lr": "training.lr",
        "device": "training.device",
        "data_root": "data.root",
        "benchmark": "data.benchmark",
        "num_workers": "data.num_workers",
        "seed": "seed",
    }
    for attr, key in mapping.items():
        value = getattr(args, attr, None)
        if value is not None:
            overrides.append(f"{key}={value}")
    return load_config(args.config, overrides)


def _cmd_make_split(args: argparse.Namespace) -> int:
    from neurolens.training.trainer import ensure_manifest
    from neurolens.utils import read_json

    config = _load_config(args)
    path = ensure_manifest(config)
    meta = read_json(path.with_name("manifest_meta.json"))
    print(f"Manifest ({meta['benchmark']}): {path.as_posix()}")
    for split, counts in meta["counts"].items():
        print(f"  {split}: {sum(counts.values())} {counts}")
    print(f"  excluded files: {len(meta['excluded'])}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from neurolens.training.trainer import train

    config = _load_config(args)
    best = train(config, resume=args.resume)
    print(f"Best checkpoint: {best.as_posix()}")
    print(
        f"Next: neurolens calibrate --run {config.experiment_name} && "
        f"neurolens evaluate --run {config.experiment_name}"
    )
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from neurolens.evaluation.evaluate import evaluate_run

    run_dir = resolve_run(args.run, Path(args.runs_dir))
    out = evaluate_run(
        run_dir,
        split=args.split,
        data_root=args.data_root,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        benchmark=not args.no_benchmark,
    )
    print(f"Evaluation written to {out.as_posix()}")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from neurolens.evaluation.evaluate import calibrate_run

    run_dir = resolve_run(args.run, Path(args.runs_dir))
    result = calibrate_run(
        run_dir,
        data_root=args.data_root,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Temperature T={result['temperature']:.4f} fitted on the validation split")
    print(
        f"  ECE before={result['ece_before']:.4f} after={result['ece_after']:.4f} "
        f"({result['ece_bins']} equal-width bins)"
    )
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    from PIL import Image

    from neurolens.inference.predictor import Predictor

    run_dir = resolve_run(args.run, Path(args.runs_dir))
    predictor = Predictor.from_run_dir(run_dir, device=args.device)
    from neurolens.data.transforms import load_image_rgb

    image = load_image_rgb(args.image)
    result, cam = predictor.explain(image)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    Image.fromarray(cam.original).save(out_dir / f"{stem}_original.png")
    Image.fromarray(cam.heatmap).save(out_dir / f"{stem}_heatmap.png")
    Image.fromarray(cam.overlay).save(out_dir / f"{stem}_overlay.png")
    from neurolens.utils import write_json

    write_json(out_dir / f"{stem}_prediction.json", result.to_dict())
    print(f"Predicted: {result.predicted_class} (raw confidence {result.raw_confidence:.3f})")
    print(f"Artifacts in {out_dir.as_posix()}")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from neurolens.optimization.onnx_export import export_and_validate

    run_dir = resolve_run(args.run, Path(args.runs_dir))
    report = export_and_validate(
        run_dir,
        opset=args.opset,
        tolerance=args.tolerance,
        quantize=args.quantize,
        data_root=args.data_root,
        n_samples=args.samples,
        iterations=args.iterations,
    )
    print(f"ONNX model: {report['onnx']['path']} ({report['onnx']['size_mb']:.2f} MB)")
    parity = report["parity"]
    verdict = "PASS" if parity["within_tolerance"] else "FAIL"
    print(
        f"  max |p_torch - p_onnx| = {parity['max_abs_prob_diff']:.2e} "
        f"(tolerance {args.tolerance:g}) -> {verdict}"
    )
    if report.get("quantized"):
        q = report["quantized"]
        if q.get("error"):
            print(f"  quantization failed: {q['error']}")
        else:
            print(
                f"  int8 dynamic: {q['size_mb']:.2f} MB, "
                f"max prob diff {q['parity']['max_abs_prob_diff']:.2e}, "
                f"top-1 agreement {q['parity']['top1_agreement']:.3f}"
            )
    return 0


def _cmd_export_web(args: argparse.Namespace) -> int:
    from neurolens.optimization.web_export import export_web_bundle

    run_dir = resolve_run(args.run, Path(args.runs_dir))
    meta = export_web_bundle(run_dir, Path(args.out), opset=args.opset)
    parity = meta["parity"]
    print(f"Web bundle for {meta['model_name']} -> {Path(args.out).as_posix()}")
    print(f"  model.onnx {meta['onnx_size_mb']} MB, temperature {meta['temperature']}")
    print(
        f"  parity on {parity['inputs']}: max |dp| {parity['max_abs_prob_diff']:.1e}, "
        f"CAM vs Grad-CAM {parity['max_abs_cam_diff_vs_gradcam']:.1e}, "
        f"top-1 agreement {parity['top1_agreement']:.3f}"
    )
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from neurolens.evaluation.benchmark import benchmark_run

    run_dir = resolve_run(args.run, Path(args.runs_dir))
    result = benchmark_run(
        run_dir,
        device=args.device,
        batch_sizes=args.batch_sizes,
        iterations=args.iterations,
        warmup=args.warmup,
    )
    for row in result["latency"]:
        print(
            f"  batch={row['batch_size']:>3} mean={row['mean_ms']:.2f} ms "
            f"p95={row['p95_ms']:.2f} ms throughput={row['throughput_img_s']:.1f} img/s"
        )
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from neurolens.evaluation.compare import build_comparison

    table = build_comparison(Path(args.runs_dir), Path(args.out))
    print(table.to_string(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neurolens", description=__doc__.split("\n")[0])
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    def run_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--run", required=True, help="run name under --runs-dir, or a run directory")
        p.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
        p.add_argument("--device", default="auto")

    def data_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data-root", default=None, help="override the data root stored in the run")
        p.add_argument("--batch-size", type=int, default=64)
        p.add_argument("--num-workers", type=int, default=2)

    def config_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", default=None, help="YAML experiment config")
        p.add_argument(
            "--set",
            action="append",
            metavar="KEY=VALUE",
            help="override any config key, e.g. --set training.lr=1e-4",
        )
        p.add_argument("--experiment-name")
        p.add_argument("--arch", choices=["custom_cnn", "resnet18", "efficientnet_b0"])
        p.add_argument("--epochs", type=int)
        p.add_argument("--batch-size", type=int)
        p.add_argument("--lr", type=float)
        p.add_argument("--device")
        p.add_argument("--data-root")
        p.add_argument("--benchmark", choices=["original", "leakage_audited"])
        p.add_argument("--num-workers", type=int)
        p.add_argument("--seed", type=int)

    p = sub.add_parser("audit", help="audit dataset integrity and duplicates")
    p.add_argument("--data-root", default="data/raw/brain-tumor-mri-dataset")
    p.add_argument("--out", default="artifacts/audit")
    p.add_argument("--threshold", type=int, default=4, help="max pHash Hamming distance")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--manifest", help="use train/val/test partitions from this manifest.csv")
    p.set_defaults(func=_cmd_audit)

    p = sub.add_parser("make-split", help="create a reproducible split manifest")
    config_args(p)
    p.set_defaults(func=_cmd_make_split)

    p = sub.add_parser("train", help="train a model")
    config_args(p)
    p.add_argument("--resume", action="store_true", help="continue from last.pt")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("calibrate", help="fit temperature scaling on the validation split")
    run_args(p)
    data_args(p)
    p.set_defaults(func=_cmd_calibrate)

    p = sub.add_parser("evaluate", help="evaluate a checkpoint and save metrics/plots")
    run_args(p)
    data_args(p)
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--no-benchmark", action="store_true", help="skip latency measurement")
    p.set_defaults(func=_cmd_evaluate)

    p = sub.add_parser("explain", help="Grad-CAM for a single image")
    run_args(p)
    p.add_argument("--image", required=True)
    p.add_argument("--out", default="artifacts/explanations")
    p.set_defaults(func=_cmd_explain)

    p = sub.add_parser("export-onnx", help="export to ONNX, validate and benchmark")
    p.add_argument("--run", required=True)
    p.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--tolerance", type=float, default=1e-4)
    p.add_argument("--quantize", action="store_true", help="also try int8 dynamic quantization")
    p.add_argument("--data-root", default=None)
    p.add_argument("--samples", type=int, default=32, help="validation images for parity checks")
    p.add_argument("--iterations", type=int, default=50)
    p.set_defaults(func=_cmd_export)

    p = sub.add_parser("export-web", help="export a PyTorch-free bundle for deploy/vercel")
    p.add_argument("--run", required=True)
    p.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    p.add_argument("--out", default="deploy/vercel/model")
    p.add_argument("--opset", type=int, default=17)
    p.set_defaults(func=_cmd_export_web)

    p = sub.add_parser("benchmark", help="measure PyTorch inference latency")
    run_args(p)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32])
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.set_defaults(func=_cmd_benchmark)

    p = sub.add_parser("compare", help="aggregate evaluated runs into a comparison table")
    p.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    p.add_argument("--out", default="artifacts/reports")
    p.set_defaults(func=_cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
