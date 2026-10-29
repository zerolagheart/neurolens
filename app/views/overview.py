"""Overview: what NeuroLens is, what is available, and how to get started."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from neurolens.data.classes import CANONICAL_CLASSES, display_name
from neurolens.inference.registry import discover_models
from neurolens.models.factory import ARCHITECTURES
from ui import API_DOCS_URL, AUDIT_DIR, RUNS_DIR, card, disclaimer, header, load_json, pill, stat

header(
    "Explainable MRI classification",
    "NeuroLens",
    "Train, evaluate, calibrate and explain image-level brain MRI classifiers on the public "
    "Nickparvar Brain Tumor MRI dataset, with reproducible splits, leakage auditing and Grad-CAM.",
)
disclaimer()

models = discover_models(RUNS_DIR)
audit = load_json(AUDIT_DIR / "audit_report.json")
evaluated = [m for m in models if m.has_test_metrics]

c1, c2, c3, c4 = st.columns(4)
with c1:
    stat(
        "Trained runs",
        str(len(models)),
        f"{len({m.architecture for m in models})} of {len(ARCHITECTURES)} architectures",
    )
with c2:
    stat("Evaluated on test", str(len(evaluated)), "offline evaluation files found")
with c3:
    stat("Classes", str(len(CANONICAL_CLASSES)), "image-level labels")
with c4:
    stat(
        "Dataset audit",
        "available" if audit else "not run",
        f"{audit['total_files']:,} files audited" if audit else "run `neurolens audit`",
    )

st.write("")
st.subheader("Supported classes")
cols = st.columns(4)
descriptions = {
    "glioma": "Dataset label for scans in the glioma folder.",
    "meningioma": "Dataset label for scans in the meningioma folder.",
    "notumor": "Dataset label for scans without a tumour annotation.",
    "pituitary": "Dataset label for scans in the pituitary folder.",
}
for col, name in zip(cols, CANONICAL_CLASSES, strict=True):
    with col:
        card(
            display_name(name),
            descriptions[name] + " Labels come from the dataset, not from "
            "clinical review in this project.",
        )

st.write("")
st.subheader("Architectures")
rows = []
for spec in ARCHITECTURES.values():
    runs = [m.name for m in models if m.architecture == spec.name]
    rows.append(
        {
            "Architecture": spec.name,
            "Description": spec.description,
            "Initialisation": "ImageNet (torchvision)"
            if spec.supports_pretrained
            else "from scratch",
            "Grad-CAM layer": spec.gradcam_layer_name,
            "Trained runs": ", ".join(runs) if runs else "not trained",
        }
    )
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

left, right = st.columns([3, 2])
with left:
    st.subheader("Pipeline")
    st.markdown(
        "1. **Audit** the dataset: counts, formats, unreadable files, SHA-256 exact duplicates "
        "and pHash near-duplicate candidates across partitions.\n"
        "2. **Split** reproducibly: a stratified validation split from `Training`; the original "
        "`Testing` set is kept for final evaluation only. An optional *leakage-audited* "
        "benchmark removes duplicate-linked test images.\n"
        "3. **Train** a custom CNN, ResNet-18 or EfficientNet-B0 with early stopping on the "
        "validation split.\n"
        "4. **Calibrate** with temperature scaling on validation data, then **evaluate** on test.\n"
        "5. **Explain** single predictions with Grad-CAM, and **export** to ONNX Runtime."
    )
with right:
    st.subheader("Status")
    st.markdown(
        pill("audit report", bool(audit))
        + pill("trained model", bool(models))
        + pill("calibration", any(m.has_calibration for m in models))
        + pill("test metrics", bool(evaluated))
        + pill("ONNX export", any(m.has_onnx for m in models)),
        unsafe_allow_html=True,
    )
    st.markdown(
        f"- API documentation (when the API is running): [{API_DOCS_URL}]({API_DOCS_URL})\n"
        "- Project documentation: see the **Documentation** page or `docs/` in the repository.\n"
        "- Dataset: [Kaggle — masoudnickparvar/brain-tumor-mri-dataset]"
        "(https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)"
    )
    if not models:
        st.info(
            "No trained model yet. Start with `neurolens train --config configs/resnet18.yaml`."
        )
