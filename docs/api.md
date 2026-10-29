# API

Start the service from the repository root:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Interactive OpenAPI docs: <http://localhost:8000/docs> (ReDoc at `/redoc`, schema at
`/openapi.json`). Models are discovered under `NEUROLENS_RUNS_DIR` and loaded **once at startup**;
restart the service after training a new model.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness (`status`) and readiness (`ready` = a model is loaded) |
| GET | `/models` | supported architectures and discovered checkpoints (no file paths) |
| POST | `/predict` | multipart image upload -> prediction |
| POST | `/explain` | prediction + Grad-CAM original/heatmap/overlay/legend as base64 PNG |
| GET | `/metrics` | saved offline evaluations, model comparison, and live process statistics |

Optional query parameter for `/predict` and `/explain`: `model=<run name from /models>`. Only names
of discovered runs are accepted; the server never opens client-supplied paths or model files.

## Status codes

| Code | When |
|---|---|
| 200 | success |
| 400 | empty, corrupt or undecodable file, or image smaller than 32 px |
| 404 | unknown `model` name |
| 413 | file larger than `NEUROLENS_MAX_UPLOAD_MB` (default 10) or too many pixels |
| 415 | decodable but unsupported format (supported: JPEG, PNG, BMP, TIFF) |
| 422 | missing `file` field |
| 503 | no trained model loaded — the API never fabricates predictions |

## Example

```bash
curl -s http://localhost:8000/health
curl -s -F "file=@data/raw/brain-tumor-mri-dataset/Testing/glioma/Te-gl_1.jpg" \
     "http://localhost:8000/predict?model=resnet18_baseline"
```

Response shape (`PredictionResponse`; values illustrative):

```json
{
  "predicted_class": "glioma",
  "predicted_label": "Glioma",
  "class_names": ["glioma", "meningioma", "notumor", "pituitary"],
  "raw_probabilities": {"glioma": 0.0, "meningioma": 0.0, "notumor": 0.0, "pituitary": 0.0},
  "calibrated_probabilities": {"glioma": 0.0, "meningioma": 0.0, "notumor": 0.0, "pituitary": 0.0},
  "raw_confidence": 0.0,
  "calibrated_confidence": 0.0,
  "uncertainty": {
    "entropy_nats": 0.0, "normalized_entropy": 0.0, "abstained": false, "reasons": [],
    "probabilities_used": "calibrated",
    "policy": {"enabled": true, "min_confidence": 0.7, "max_normalized_entropy": 0.6},
    "top_probability": 0.0
  },
  "model": {"name": "resnet18_baseline", "version": "resnet18_baseline-e007",
            "architecture": "resnet18", "benchmark": "original", "temperature": 1.0},
  "preprocessing": {"image_size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
  "warnings": [],
  "disclaimer": "Research and educational use only. NeuroLens is not a medical device ..."
}
```

`/explain` returns `{"prediction": PredictionResponse, "explanation": {...}}` where `explanation`
has `target_class`, `target_layer`, `degenerate`, `original_png_base64`, `heatmap_png_base64`,
`overlay_png_base64`, `legend_png_base64`, `legend` and `interpretation_note`.

Python client:

```python
import httpx

with open("scan.jpg", "rb") as f:
    r = httpx.post("http://localhost:8000/predict", files={"file": ("scan.jpg", f, "image/jpeg")})
r.raise_for_status()
print(r.json()["predicted_label"], r.json()["uncertainty"]["abstained"])
```

## Configuration (environment variables)

See `.env.example`. Key settings: `NEUROLENS_RUNS_DIR`, `NEUROLENS_DEFAULT_MODEL`,
`NEUROLENS_DEVICE` (`cpu`, `cuda`, `auto`), `NEUROLENS_MAX_UPLOAD_MB`, `NEUROLENS_MAX_MODELS`,
`NEUROLENS_CORS_ORIGINS` (comma-separated), `NEUROLENS_ABSTENTION_ENABLED`,
`NEUROLENS_MIN_CONFIDENCE`, `NEUROLENS_MAX_NORMALIZED_ENTROPY`, `NEUROLENS_LOG_LEVEL`,
`NEUROLENS_LOG_JSON` (`1` for JSON log lines).

Logs record endpoint, model, byte count, outcome and latency — never image content or the
client's filename.
