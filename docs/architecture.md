# Architecture

```mermaid
flowchart LR
    subgraph Data["src/neurolens/data"]
        D1[discovery.py<br/>validate Training/Testing] --> D2[audit.py<br/>SHA-256 + pHash]
        D1 --> D3[splits.py<br/>manifest.csv]
        D2 -->|duplicate groups| D3
        D4[transforms.py<br/>PreprocessConfig]
    end
    subgraph Train["src/neurolens/training"]
        T1[config.py<br/>YAML + overrides] --> T2[trainer.py<br/>AMP, early stopping, resume]
        T2 --> T3[checkpoint.py<br/>best.pt + metadata]
    end
    subgraph Eval["src/neurolens/evaluation"]
        E1[metrics.py] --> E3[evaluate.py]
        E2[calibration.py<br/>temperature, ECE] --> E3
        E4[benchmark.py] --> E3
        E3 --> E5[compare.py]
    end
    subgraph Serve["inference + explainability"]
        I1[predictor.py] --> X1[gradcam.py]
        I2[registry.py]
    end
    O1[optimization/onnx_export.py]
    D3 --> T2
    D4 --> T2
    T3 --> E3
    T3 --> I1
    T3 --> O1
    I1 --> S1[api/service.py]
    I2 --> S1
    S1 --> A1[FastAPI routes]
    S1 --> U1[Streamlit app<br/>in-process]
    A1 -.HTTP when NEUROLENS_API_URL is set.-> U2[Streamlit app<br/>API mode]
```

## Design decisions

* **`src/` layout with one importable package (`neurolens`).** The prompt's `src/data`,
  `src/models`, … structure is kept, but nested inside `src/neurolens/` so the code is importable
  as `neurolens.*` (installing a top-level package literally named `src` would collide with other
  projects). `api/` and `app/` stay at the repository root as deployable applications.
* **The checkpoint is the contract.** `best.pt` carries architecture, class names, preprocessing,
  configuration and software versions; inference, evaluation and export all rebuild from it and
  validate it. It is loaded with `torch.load(weights_only=True)`.
* **One service, two front doors.** `api/service.py` produces the Pydantic responses for both the
  FastAPI routes and the Streamlit page, so the UI cannot drift from the API contract.
* **Files over infrastructure.** Runs, metrics and reports are JSON/CSV/PNG files under
  `artifacts/`. No database, tracking server or message queue is required.
* **Honest absence.** Missing checkpoints, evaluations or benchmarks are reported as unavailable
  everywhere (CLI, API 503, UI empty states, `not trained` rows). Nothing substitutes random
  weights or placeholder numbers.
* **No OpenCV.** Pillow, NumPy and SciPy cover decoding, resizing, the DCT for perceptual hashing
  and colour mapping, so the heavyweight native dependency is unnecessary.
