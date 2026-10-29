"""ONNX export/validation; skipped when the optional ``onnx`` extra is not installed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from neurolens.optimization.onnx_export import (
    compare_outputs,
    export_and_validate,
    export_onnx,
    ort_session,
)
from neurolens.training.checkpoint import model_from_checkpoint

pytestmark = pytest.mark.onnx


def test_export_and_parity(tiny_checkpoint: Path, tmp_path: Path) -> None:
    model, _ = model_from_checkpoint(tiny_checkpoint / "best.pt")
    path = export_onnx(model, 64, tmp_path / "m.onnx")
    session = ort_session(path)
    x = np.random.default_rng(0).standard_normal((3, 3, 64, 64)).astype(np.float32)
    parity = compare_outputs(model, session, x, tolerance=1e-4)
    assert parity["within_tolerance"] and parity["top1_agreement"] == 1.0
    strict = compare_outputs(model, session, x, tolerance=0.0)
    assert strict["within_tolerance"] == (strict["max_abs_prob_diff"] == 0.0)


def test_export_and_validate_report(tiny_checkpoint: Path) -> None:
    report = export_and_validate(
        tiny_checkpoint, quantize=True, n_samples=4, iterations=3, batch_sizes=[1]
    )
    assert (tiny_checkpoint / "onnx" / "onnx_report.json").exists()
    assert "random" in report["parity_inputs"]  # dataset is unavailable in the fixture
    assert report["benchmark"]["onnx"][0]["batch_size"] == 1
    assert "quantized" in report  # either measured results or a reported error
