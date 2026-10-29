"""Inference latency and throughput measurement.

Procedure (recorded with every result so measurements are only compared like-for-like):

* model in ``eval()`` mode under ``torch.inference_mode()``, float32, no AMP;
* input is a pre-built normalised tensor, i.e. **image decoding and preprocessing are excluded**
  (measured separately as ``preprocess_ms`` when a sample image is available);
* ``warmup`` untimed iterations, then ``iterations`` timed iterations with
  ``time.perf_counter``; on CUDA, ``torch.cuda.synchronize()`` is called before reading the
  clock on both sides of every iteration;
* throughput = ``batch_size / mean latency``.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from neurolens.data.transforms import PreprocessConfig, load_image_rgb, preprocess_image
from neurolens.models.factory import count_parameters
from neurolens.utils import device_description, file_size_mb, resolve_device, utc_now, write_json

MEASUREMENT_PROCEDURE = (
    "eval mode, torch.inference_mode, float32; preprocessing excluded; warm-up iterations "
    "discarded; CUDA synchronised around each timed iteration; throughput = batch / mean latency"
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summarize_timings(timings_s: list[float], batch_size: int) -> dict[str, float]:
    arr = np.asarray(timings_s) * 1000.0
    mean = float(arr.mean())
    return {
        "batch_size": batch_size,
        "mean_ms": mean,
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "std_ms": float(arr.std()),
        "throughput_img_s": batch_size / (mean / 1000.0),
    }


def measure_latency(
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    image_size: int,
    iterations: int = 50,
    warmup: int = 10,
) -> dict[str, float]:
    model.eval()
    inputs = torch.randn(batch_size, 3, image_size, image_size, device=device)
    timings: list[float] = []
    with torch.inference_mode():
        for i in range(warmup + iterations):
            _sync(device)
            start = time.perf_counter()
            model(inputs)
            _sync(device)
            if i >= warmup:
                timings.append(time.perf_counter() - start)
    return summarize_timings(timings, batch_size)


def measure_preprocessing(image_bytes: bytes, cfg: PreprocessConfig, iterations: int = 30) -> float:
    """Mean milliseconds to decode an encoded image and build the model input tensor."""
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        preprocess_image(load_image_rgb(io.BytesIO(image_bytes)), cfg)
        timings.append(time.perf_counter() - start)
    return float(np.mean(timings) * 1000.0)


def benchmark_model(
    model: nn.Module,
    device: torch.device,
    preprocess: PreprocessConfig,
    batch_sizes: list[int],
    iterations: int = 50,
    warmup: int = 10,
    sample_image: Path | None = None,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "measured_at": utc_now(),
        "runtime": "pytorch",
        "device": device_description(device),
        "device_type": device.type,
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "dtype": "float32",
        "image_size": preprocess.image_size,
        "warmup_iterations": warmup,
        "timed_iterations": iterations,
        "preprocessing_included": False,
        "procedure": MEASUREMENT_PROCEDURE,
        "parameters": count_parameters(model),
        "latency": [
            measure_latency(model, device, bs, preprocess.image_size, iterations, warmup)
            for bs in batch_sizes
        ],
    }
    if checkpoint is not None:
        result["checkpoint_size_mb"] = file_size_mb(checkpoint)
    if sample_image is not None and Path(sample_image).exists():
        result["preprocess_ms_single_image"] = measure_preprocessing(
            Path(sample_image).read_bytes(), preprocess
        )
    return result


def benchmark_run(
    run_dir: Path,
    device: str = "auto",
    batch_sizes: list[int] | None = None,
    iterations: int = 50,
    warmup: int = 10,
) -> dict[str, Any]:
    """Benchmark a trained run's best checkpoint and save ``benchmarks/pytorch_<device>.json``."""
    from neurolens.training.checkpoint import model_from_checkpoint, preprocess_from_metadata

    torch_device = resolve_device(device)
    model, metadata = model_from_checkpoint(run_dir / "best.pt", torch_device)
    result = benchmark_model(
        model,
        torch_device,
        preprocess_from_metadata(metadata),
        batch_sizes or [1, 32],
        iterations,
        warmup,
        checkpoint=run_dir / "best.pt",
    )
    write_json(Path(run_dir) / "benchmarks" / f"pytorch_{torch_device.type}.json", result)
    return result
