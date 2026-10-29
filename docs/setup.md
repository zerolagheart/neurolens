# Dataset setup

NeuroLens uses the **Brain Tumor MRI Dataset** by Masoud Nickparvar:
<https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset>.
The project never downloads the dataset itself and never asks for Kaggle credentials. Review and
respect the dataset's licence and terms on Kaggle before use.

## Download

**Option A — browser:** sign in to Kaggle, open the dataset page, click *Download*, and extract
the archive.

**Option B — Kaggle CLI** (requires your own `~/.kaggle/kaggle.json` API token):

```bash
pip install kaggle
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data/raw
```

Then extract so the layout looks like this (an extra wrapper directory is fine; discovery searches
two levels deep):

```
data/raw/brain-tumor-mri-dataset/
├── Training/
│   ├── glioma/  meningioma/  notumor/  pituitary/
└── Testing/
    ├── glioma/  meningioma/  notumor/  pituitary/
```

Windows PowerShell:

```powershell
Expand-Archive brain-tumor-mri-dataset.zip -DestinationPath data/raw/brain-tumor-mri-dataset
```

Linux / macOS:

```bash
unzip brain-tumor-mri-dataset.zip -d data/raw/brain-tumor-mri-dataset
```

Class folders are matched case-insensitively, and common aliases (`no_tumor`,
`pituitary_tumor`, …) are accepted. Discovery fails with a clear message if a split or class
folder is missing or ambiguous. `data/` is excluded from Git.

The Kaggle dataset has been published in more than one revision (file counts differ between
versions). The audit report records exactly what you have locally.

## Audit before training

```bash
neurolens audit --data-root data/raw/brain-tumor-mri-dataset
```

This writes `artifacts/audit/audit_report.json`, `audit_files.csv` (per-file hashes and
properties) and `duplicate_pairs.csv`. It reads files only; nothing is moved or deleted.

## Splits

```bash
neurolens make-split --config configs/resnet18.yaml                    # original benchmark
neurolens make-split --config configs/resnet18_leakage_audited.yaml    # modified benchmark
```

* **original**: `Testing/` is the untouched test set. 15 % of `Training/` (stratified, seed 42)
  becomes the validation set used for early stopping, model selection and calibration.
* **leakage_audited**: duplicate groups (exact + near-duplicate links from the audit) never
  straddle the train/validation boundary; test images whose group contains any training image are
  removed, and only one image per duplicate group is kept inside the test set. Every exclusion is
  listed in `manifest_meta.json`. This is a different benchmark: report it separately.

Manifests store paths relative to the dataset root, so they are portable between machines.
Training builds the configured manifest automatically if it does not exist yet.
