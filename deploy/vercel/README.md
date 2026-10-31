# NeuroLens web demo on Vercel

A Vercel-ready version of NeuroLens: a static page (`public/index.html`) plus one Python
serverless function (`api/index.py`) that runs the exported model with **ONNX Runtime**.

Why not the Streamlit app? Streamlit needs a long-running server with WebSockets, and PyTorch
is far larger than Vercel's ~250 MB limit for an unpacked Python function. This bundle has
no PyTorch: inference uses ONNX Runtime and NumPy, and Grad-CAM is computed exactly from the
exported feature maps and final-layer weights (see `src/neurolens/optimization/web_export.py`).

```
deploy/vercel/
├── api/index.py        FastAPI app: GET /api/health, GET /api/info, POST /api/analyze
├── public/index.html   the page (upload, prediction, uncertainty, Grad-CAM, downloads)
├── model/              exported bundle: model.onnx, cam_weights.npy, colormap.npy, meta.json
├── requirements.txt    function dependencies (fastapi, numpy, pillow, onnxruntime)
├── vercel.json         routes /api/* to the function, bundles model/ with it
└── dev_server.py       local preview (not deployed)
```

## Refresh the bundled model (optional)

The repository ships EfficientNet-B0 (`efficientnet_b0_baseline`, test macro-F1 0.953 on the
original split). To export a different run, from the repository root:

```bash
neurolens export-web --run efficientnet_b0_baseline --out deploy/vercel/model
```

The export refuses to write a bundle unless the ONNX probabilities match PyTorch to within
1e-4 and the NumPy Grad-CAM matches autograd Grad-CAM to within 1e-3, checked on validation
images.

## Preview locally

```bash
python deploy/vercel/dev_server.py        # http://localhost:3000
```

## Deploy from GitHub (recommended)

1. Sign in at <https://vercel.com> with the GitHub account that owns the repository.
2. **Add New… → Project**, then import `zerolagheart/neurolens`. Grant the Vercel GitHub app
   access to that repository if asked.
3. On the configuration screen:
   * **Root Directory:** click *Edit* and choose `deploy/vercel`. This is the important step,
     because the repository root is the full PyTorch project.
   * **Framework Preset:** *Other*.
   * Leave the build and output settings empty; `vercel.json` sets the output to `public`.
   * **Environment variables:** none are required. Optionally set `NEUROLENS_MAX_UPLOAD_MB`
     (default 4; keep it under Vercel's 4.5 MB request limit).
4. Click **Deploy**. When the build finishes you get `https://<project>.vercel.app`.
5. Check `https://<project>.vercel.app/api/health`, which should return `"ready": true`, then
   open the site and analyse an image from the dataset's `Testing` folder.

Every push to `main` triggers a production deployment, and pull requests get preview URLs. To
skip builds for commits that don't touch the demo, enable *Settings → Git → Ignored Build Step*
(or "skip deployments when there are no changes to the root directory").

## Deploy from your machine (alternative)

```bash
npm i -g vercel
cd deploy/vercel
vercel           # first run: log in, link or create the project, accept the detected settings
vercel --prod    # promote to production
```

When deploying this way, `deploy/vercel` is the project root, so no Root Directory setting is
needed.

## Limits and notes

* **Cold starts:** the first request after an idle period loads ONNX Runtime and the 15 MB
  model, which takes a few seconds. Later requests take well under a second on Vercel's CPUs.
* **Uploads** go to the function as `multipart/form-data` and are processed in memory, never
  stored. The function logs only byte counts and latency.
* **Weights are public** once the repository is public. Confirm that the dataset's licence on
  Kaggle (and its original sources) permits redistributing a model trained on it.
* The page states the research-only disclaimer. The metrics it shows come from the original
  test split, which the project's audit found to be heavily linked to the training data.
