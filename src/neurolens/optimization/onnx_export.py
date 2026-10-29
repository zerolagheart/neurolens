"""ONNX export, ONNX Runtime validation, CPU benchmarking and optional dynamic quantization.

All numbers in the report are measured on this machine; quantization is reported as-is,
including when it is slower, larger or less accurate than the float model.
"""

from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from neurolens.evaluation.benchmark import MEASUREMENT_PROCEDURE, summarize_timings
from neurolens.evaluation.metrics import softmax
from neurolens.training.checkpoint import model_from_checkpoint, preprocess_from_metadata
from neurolens.utils import device_description, file_size_mb, get_logger, utc_now, write_json

log = get_logger("onnx")


class OptionalDependencyError(ImportError):
    pass


def _require_onnx() -> tuple[Any, Any]:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise OptionalDependencyError(
            'ONNX export needs the optional dependencies: pip install -e ".[onnx]"'
        ) from exc
    return onnx, ort


def export_onnx(model: nn.Module, image_size: int, path: Path, opset: int = 17) -> Path:
    """Export with a dynamic batch dimension using the TorchScript-based exporter.

    The legacy exporter (``dynamo=False``) is used because it is available across the supported
    PyTorch versions and handles these CNNs without extra dependencies.
    """
    onnx, _ = _require_onnx()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    dummy = torch.randn(1, 3, image_size, image_size)
    kwargs: dict[str, Any] = {
        "input_names": ["input"],
        "output_names": ["logits"],
        "dynamic_axes": {"input": {0: "batch"}, "logits": {0: "batch"}},
        "opset_version": opset,
        "do_constant_folding": True,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    torch.onnx.export(model, (dummy,), str(path), **kwargs)
    onnx.checker.check_model(onnx.load(str(path)))
    return path


def ort_session(path: Path, threads: int | None = None) -> Any:
    _, ort = _require_onnx()
    options = ort.SessionOptions()
    if threads:
        options.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def compare_outputs(
    model: nn.Module, session: Any, inputs: np.ndarray, tolerance: float
) -> dict[str, Any]:
    """Compare PyTorch (CPU, float32) and ONNX Runtime softmax outputs on identical inputs."""
    with torch.inference_mode():
        torch_logits = model.cpu().eval()(torch.from_numpy(inputs)).numpy()
    ort_logits = session.run(None, {"input": inputs})[0]
    p_torch, p_ort = softmax(torch_logits), softmax(ort_logits)
    max_diff = float(np.abs(p_torch - p_ort).max())
    return {
        "n_inputs": len(inputs),
        "max_abs_prob_diff": max_diff,
        "max_abs_logit_diff": float(np.abs(torch_logits - ort_logits).max()),
        "top1_agreement": float((p_torch.argmax(1) == p_ort.argmax(1)).mean()),
        "tolerance": tolerance,
        "within_tolerance": max_diff <= tolerance,
    }


def benchmark_session(
    session: Any, image_size: int, batch_sizes: list[int], iterations: int, warmup: int = 10
) -> list[dict[str, float]]:
    rows = []
    for bs in batch_sizes:
        x = (
            np.random.default_rng(0)
            .standard_normal((bs, 3, image_size, image_size))
            .astype(np.float32)
        )
        timings = []
        for i in range(warmup + iterations):
            start = time.perf_counter()
            session.run(None, {"input": x})
            if i >= warmup:
                timings.append(time.perf_counter() - start)
        rows.append(summarize_timings(timings, bs))
    return rows


def benchmark_torch_cpu(
    model: nn.Module, image_size: int, batch_sizes: list[int], iterations: int, warmup: int = 10
) -> list[dict[str, float]]:
    from neurolens.evaluation.benchmark import measure_latency

    return [
        measure_latency(model.cpu(), torch.device("cpu"), bs, image_size, iterations, warmup)
        for bs in batch_sizes
    ]


def quantize_dynamic_int8(src: Path, dst: Path) -> Path:
    """ONNX Runtime dynamic (weight-only int8, activations quantised at runtime) quantization.

    Only operator types supported by ``quantize_dynamic`` are converted; for convolutional
    networks most compute stays in Conv layers, so gains vary and may be negative.
    """
    _require_onnx()
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = dst.with_name(dst.stem + "_pre.onnx")
    quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
    quantize_dynamic(str(pre), str(dst), weight_type=QuantType.QUInt8)
    pre.unlink(missing_ok=True)
    return dst


def _sample_inputs(
    run_dir: Path, metadata: dict[str, Any], data_root: str | None, n: int
) -> tuple[np.ndarray, str]:
    """Real validation images when available, otherwise seeded random tensors."""
    from neurolens.data.splits import load_manifest
    from neurolens.data.transforms import load_image_rgb, preprocess_image

    preprocess = preprocess_from_metadata(metadata)
    root = Path(data_root or metadata["experiment"]["data"]["root"])
    manifest = run_dir / "manifest.csv"
    if manifest.exists() and root.exists():
        try:
            frame, _ = load_manifest(manifest, root)
        except FileNotFoundError:
            frame = None
        if frame is not None:
            val = frame[frame["split"] == "val"].sample(
                n=min(n, (frame["split"] == "val").sum()), random_state=0
            )
            batch = torch.cat(
                [preprocess_image(load_image_rgb(p), preprocess) for p in val["path"]]
            )
            return batch.numpy().astype(np.float32), f"{len(val)} validation images"
    rng = np.random.default_rng(0)
    shape = (n, 3, preprocess.image_size, preprocess.image_size)
    return rng.standard_normal(shape).astype(
        np.float32
    ), f"{n} seeded random tensors (dataset unavailable)"


def export_and_validate(
    run_dir: Path,
    opset: int = 17,
    tolerance: float = 1e-4,
    quantize: bool = False,
    data_root: str | None = None,
    n_samples: int = 32,
    iterations: int = 50,
    batch_sizes: list[int] | None = None,
) -> dict[str, Any]:
    """Export ``run_dir/best.pt`` and write ``run_dir/onnx/onnx_report.json``."""
    _require_onnx()
    run_dir = Path(run_dir)
    batch_sizes = batch_sizes or [1, 8]
    model, metadata = model_from_checkpoint(run_dir / "best.pt", "cpu")
    image_size = preprocess_from_metadata(metadata).image_size
    out_dir = run_dir / "onnx"
    onnx_path = export_onnx(model, image_size, out_dir / "model.onnx", opset)
    session = ort_session(onnx_path)
    inputs, input_desc = _sample_inputs(run_dir, metadata, data_root, n_samples)

    report: dict[str, Any] = {
        "generated_at": utc_now(),
        "model_version": metadata.get("model_version"),
        "architecture": metadata["architecture"],
        "opset": opset,
        "exporter": "torch.onnx.export (TorchScript-based), dynamic batch axis",
        "parity_inputs": input_desc,
        "onnx": {"path": f"{run_dir.name}/onnx/model.onnx", "size_mb": file_size_mb(onnx_path)},
        "pytorch_checkpoint_mb": file_size_mb(run_dir / "best.pt"),
        "parity": compare_outputs(model, session, inputs, tolerance),
        "benchmark_conditions": {
            "device": device_description(torch.device("cpu")),
            "runtime_versions": {
                "torch": torch.__version__,
                "onnxruntime": _require_onnx()[1].__version__,
            },
            "batch_sizes": batch_sizes,
            "iterations": iterations,
            "procedure": MEASUREMENT_PROCEDURE + "; ONNX Runtime CPUExecutionProvider",
        },
        "benchmark": {
            "pytorch_cpu": benchmark_torch_cpu(model, image_size, batch_sizes, iterations),
            "onnx": benchmark_session(session, image_size, batch_sizes, iterations),
        },
    }

    if quantize:
        q_path = out_dir / "model_int8_dynamic.onnx"
        try:
            quantize_dynamic_int8(onnx_path, q_path)
            q_session = ort_session(q_path)
            report["quantized"] = {
                "method": "onnxruntime.quantization.quantize_dynamic (QUInt8 weights)",
                "path": f"{run_dir.name}/onnx/{q_path.name}",
                "size_mb": file_size_mb(q_path),
                "parity": compare_outputs(model, q_session, inputs, tolerance),
                "benchmark": benchmark_session(q_session, image_size, batch_sizes, iterations),
            }
        except Exception as exc:  # report any quantizer/runtime failure instead of hiding it
            log.warning("Dynamic quantization failed: %s", exc)
            report["quantized"] = {"error": f"{type(exc).__name__}: {exc}"}

    write_json(out_dir / "onnx_report.json", report)
    return report
