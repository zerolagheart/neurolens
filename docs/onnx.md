# ONNX export and optimisation

Requires the optional extra: `pip install -e ".[onnx]"`.

```bash
neurolens export-onnx --run resnet18_baseline                 # export + validate + benchmark
neurolens export-onnx --run resnet18_baseline --quantize      # also try int8 dynamic quantization
neurolens export-onnx --run resnet18_baseline --tolerance 1e-5 --samples 64 --iterations 100
```

What happens:

1. The best checkpoint is exported with `torch.onnx.export` (TorchScript-based exporter, opset 17
   by default, dynamic batch axis, constant folding) and checked with `onnx.checker`.
2. An ONNX Runtime `CPUExecutionProvider` session is created.
3. **Parity:** PyTorch (CPU, float32) and ONNX Runtime are run on the same inputs — up to
   `--samples` real validation images when the dataset is available, otherwise seeded random
   tensors (the report says which). The report gives the maximum absolute probability
   difference, maximum logit difference, top-1 agreement, and whether the probability difference
   is within `--tolerance` (default `1e-4`).
4. **Benchmark:** PyTorch CPU and ONNX Runtime latency for batch sizes 1 and 8 on the same machine,
   with the same warm-up/timing procedure as `neurolens benchmark`.
5. **Quantization (optional):** `onnxruntime.quantization.quantize_dynamic` with uint8 weights
   after ONNX Runtime's pre-processing pass. Dynamic quantization only converts supported operator
   types; CNNs spend most time in convolutions, so it may not help and can be slower. Size,
   latency, parity and top-1 agreement are reported as measured — regressions included. If the
   quantizer fails, the error is recorded in the report.

Output: `artifacts/runs/<run>/onnx/model.onnx`, `model_int8_dynamic.onnx` (if requested) and
`onnx_report.json`, which the Model Laboratory renders.

If no trained checkpoint exists, the command exits with instructions to train first; it never
reports a validation it did not run.
