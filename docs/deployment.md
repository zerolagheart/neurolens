# Running the Streamlit app and publishing a live demo

## 1. Run Streamlit locally

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1      Linux/macOS: source .venv/bin/activate
pip install -e ".[all]"
streamlit run app/streamlit_app.py            # http://localhost:8501
```

The app reads everything from `artifacts/`. A page that has nothing to show displays an empty
state with the command that produces its data:

| Page | Needs |
|---|---|
| Overview, Documentation | nothing |
| Dataset Explorer | `neurolens audit --data-root data/raw/brain-tumor-mri-dataset` |
| MRI Analysis | a trained run (`artifacts/runs/<run>/best.pt` + `best_meta.json`) |
| Model Laboratory | `neurolens calibrate` + `neurolens evaluate` for each run |

Useful settings (environment variables, see `.env.example`):

| Variable | Effect |
|---|---|
| `NEUROLENS_RUNS_DIR`, `NEUROLENS_AUDIT_DIR`, `NEUROLENS_SPLITS_DIR` | where the app looks for artifacts |
| `NEUROLENS_DATA_ROOT` | enables duplicate-pair image previews in the Dataset Explorer |
| `NEUROLENS_DEFAULT_MODEL` | run selected by default on the MRI Analysis page |
| `NEUROLENS_DEVICE` | `cpu`, `cuda` or `auto` for in-process inference |
| `NEUROLENS_API_URL` | if set, the app calls the FastAPI service instead of loading models itself |

The theme lives in `.streamlit/config.toml`, and the upload limit is `server.maxUploadSize`
(MB).

**Two-process mode (UI + API):**

```bash
uvicorn api.main:app --port 8000
# in a second terminal
NEUROLENS_API_URL=http://localhost:8000 streamlit run app/streamlit_app.py
```

`docker compose up` does the same in containers.

## 2. What a public deployment needs

Trained checkpoints and reports are git-ignored, so a host that clones this repository has
**no model**. The site will then run, but MRI Analysis will show "No trained model is
available". To get a working demo, ship one small **artifact bundle** to the host:

```
artifacts/
├── runs/<run>/best.pt, best_meta.json, calibration.json, history.csv,
│              eval/test/metrics.json, benchmarks/*.json, onnx/onnx_report.json
├── audit/audit_report.json, duplicate_pairs.csv
└── splits/*/manifest_meta.json
```

One model is enough for a demo. EfficientNet-B0 (15.6 MB checkpoint) had the best test
macro-F1 in our run. The custom CNN (4.5 MB) is the lightest.

Where to keep the bundle:

* **GitHub Release asset** (simplest): zip the folder, create a release, and attach the zip.
  The host downloads it at startup.
* **Hugging Face Hub** model repository: suited to model files and versioning.
* **Git LFS** in the repo: works, but LFS bandwidth quotas apply.

Before publishing weights, check the dataset's licence on Kaggle (and its original sources)
to confirm that redistributing a model trained on it is allowed.

## 3. Hosting options

### Vercel (static page + serverless ONNX function)

Vercel cannot host the Streamlit app, which needs a long-running WebSocket server and PyTorch.
`deploy/vercel/` is a separate, PyTorch-free web demo built for Vercel: a static page plus one
Python function that runs the exported model with ONNX Runtime and computes Grad-CAM exactly
from the exported feature maps. The repository already includes the EfficientNet-B0 bundle.
In Vercel, import the repository and set **Root Directory** to `deploy/vercel`. Full steps are
in [deploy/vercel/README.md](../deploy/vercel/README.md).

### Streamlit Community Cloud (free, simplest for the full Streamlit app)

1. Push the repository to GitHub (done: `zerolagheart/neurolens`).
2. Add a `requirements.txt` at the repo root. Community Cloud installs from it, and the CPU
   wheel index keeps PyTorch small:

   ```text
   --extra-index-url https://download.pytorch.org/whl/cpu
   torch
   torchvision
   .[app,api]
   ```

3. Make the artifact bundle available (Section 2). Download it on first start if
   `artifacts/runs` is empty, or commit a tiny demo bundle.
4. At <https://share.streamlit.io> → **Create app** → choose the repository, branch `main`,
   main file `app/streamlit_app.py`, and Python 3.11 or 3.12 (under advanced settings).
5. Put any settings in the app's **Secrets** as TOML, e.g.
   `NEUROLENS_DEFAULT_MODEL = "efficientnet_b0_baseline"` and `NEUROLENS_DEVICE = "cpu"`.
   Streamlit exposes root-level secrets as environment variables.
6. Deploy. You get a `https://<name>.streamlit.app` URL. Free apps sleep after inactivity and
   have limited memory, so keep to one model.

### Hugging Face Spaces (more memory, good for ML demos)

* Create a Space with the **Docker** SDK (and reuse this repository's `Dockerfile`), or the
  Streamlit option if it is offered.
* Spaces can store the checkpoint in the Space's own repository via Git LFS, so no download
  step is needed.
* The free CPU hardware has more RAM than Community Cloud and is comfortable for PyTorch.
* A Docker Space must listen on the port declared in its README metadata (`app_port`).
  Run Streamlit on that port: `streamlit run app/streamlit_app.py --server.port=7860
  --server.address=0.0.0.0`.

### Any Docker host (Render, Railway, Fly.io, a VPS)

```bash
docker compose up -d        # API on :8000, Streamlit on :8501
```

Mount or bake in `artifacts/`, put a reverse proxy with HTTPS in front, and expose only the
Streamlit port publicly unless you want the API to be public too.

## 4. Before going public

* Keep the research-only disclaimer visible; it is on every page by default.
* Ask visitors not to upload identifiable patient images. The app processes uploads in
  memory and does not store them, but the hosting platform's logs and terms still apply.
* State in the README which model and benchmark the demo runs, and that its metrics come
  from the original (leakage-affected) test split.
* Free tiers sleep and have cold starts. Loading PyTorch plus a model takes a few seconds on
  the first visit.
