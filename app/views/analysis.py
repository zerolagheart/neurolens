"""MRI Analysis: upload an image, run a trained model, inspect probabilities and Grad-CAM."""

from __future__ import annotations

import base64
import hashlib
import io
import json

import plotly.graph_objects as go
import streamlit as st
from PIL import Image, UnidentifiedImageError

from backend import BackendError, get_backend
from neurolens.data.classes import display_name
from ui import SERIES, disclaimer, empty_state, header, style_figure

header(
    "Inference",
    "MRI Analysis",
    "Upload a brain MRI slice, choose a trained model, and inspect its class probabilities, "
    "uncertainty and Grad-CAM explanation.",
)
disclaimer()

try:
    backend = get_backend()
    models = backend.models()
    health = backend.health()
except BackendError as exc:
    st.error(str(exc))
    st.stop()

loaded = [m for m in models.models if m.loaded]
if not health.ready or not loaded:
    empty_state(
        "No trained model is available",
        "Predictions are disabled until a checkpoint trained by this project exists. NeuroLens "
        "never substitutes random weights. Download the dataset (see Documentation), then train "
        "and calibrate a model:",
        "neurolens train --config configs/resnet18.yaml\n"
        "neurolens calibrate --run resnet18_baseline\n"
        "neurolens evaluate --run resnet18_baseline",
    )
    if health.load_errors:
        st.warning(
            "Models that failed to load: "
            + "; ".join(f"{k}: {v}" for k, v in health.load_errors.items())
        )
    if backend.mode == "in-process" and st.button("Rescan for trained models"):
        get_backend.clear()
        st.rerun()
    st.stop()

names = [m.name for m in loaded]
default_index = names.index(models.default_model) if models.default_model in names else 0
col_model, col_upload = st.columns([1, 2])
with col_model:
    model_name = st.selectbox(
        "Model", names, index=default_index, help="Trained runs discovered under artifacts/runs"
    )
    info = next(m for m in loaded if m.name == model_name)
    st.caption(
        f"{info.architecture} · epoch {info.epoch} · benchmark `{info.benchmark}` · "
        f"{'calibrated' if info.has_calibration else 'not calibrated'} · backend: {backend.mode}"
    )
with col_upload:
    upload = st.file_uploader(
        "MRI image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        help="Single 2-D slice. Max 10 MB. The file is processed in memory and not stored.",
    )

if upload is None:
    st.info(
        "Upload an image to begin. `Training` images were seen by the model; `Testing` images "
        "were not, although the dataset audit links many of them to near-duplicates in Training."
    )
    st.stop()

data = upload.getvalue()
key = hashlib.sha256(data + model_name.encode()).hexdigest()
preview, action = st.columns([1, 2])
with preview:
    try:  # re-encode for display so TIFF/BMP previews work in every browser
        st.image(
            Image.open(io.BytesIO(data)).convert("RGB"), caption="Uploaded image", width="stretch"
        )
    except (UnidentifiedImageError, OSError):
        st.warning("This file cannot be previewed; analysis will report whether it can be decoded.")
with action:
    st.write("")
    run = st.button("Run analysis", type="primary", icon=":material/play_arrow:")
    if run:
        with st.spinner("Running inference and Grad-CAM…"):
            try:
                st.session_state["analysis"] = {
                    "key": key,
                    "result": backend.explain(data, upload.name, model_name),
                }
            except BackendError as exc:
                st.session_state.pop("analysis", None)
                st.error(str(exc))
                st.stop()

state = st.session_state.get("analysis")
if not state or state["key"] != key:
    st.stop()

response = state["result"]
pred, exp = response.prediction, response.explanation
unc = pred.uncertainty

st.divider()
left, right = st.columns([2, 3])
with left:
    st.markdown('<div class="nl-stat-label">Predicted dataset class</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="nl-pred">{pred.predicted_label}</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Raw score", f"{pred.raw_confidence:.3f}", help="raw confidence: max softmax(logits)")
    m2.metric(
        "Calibrated",
        f"{pred.calibrated_confidence:.3f}" if pred.calibrated_confidence is not None else "—",
        help="max softmax(logits / T), T fitted on validation data",
    )
    m3.metric(
        "Norm. entropy",
        f"{unc.normalized_entropy:.3f}",
        help="0 = certain, 1 = uniform over classes",
    )
    if unc.abstained:
        st.warning(
            "**Uncertainty warning — prediction not accepted by the configured policy.**\n\n"
            + "\n".join(f"- {r}" for r in unc.reasons)
            + "\n\nThresholds are configuration choices, not clinically validated safety limits."
        )
    else:
        st.success(
            f"Accepted by the configured policy (min confidence "
            f"{unc.policy['min_confidence']:.2f} on {unc.probabilities_used} probabilities). "
            "A high score is not a guarantee of correctness."
        )
    for warning in pred.warnings:
        st.caption(f"⚠ {warning}")
with right:
    classes = pred.class_names
    fig = go.Figure()
    fig.add_bar(
        x=[display_name(c) for c in classes],
        y=[pred.raw_probabilities[c] for c in classes],
        name="raw softmax",
        marker_color=SERIES[0],
        hovertemplate="%{x}<br>raw: %{y:.3f}<extra></extra>",
    )
    if pred.calibrated_probabilities:
        fig.add_bar(
            x=[display_name(c) for c in classes],
            y=[pred.calibrated_probabilities[c] for c in classes],
            name=f"calibrated (T={pred.model.temperature:.2f})",
            marker_color=SERIES[1],
            hovertemplate="%{x}<br>calibrated: %{y:.3f}<extra></extra>",
        )
    fig.update_yaxes(range=[0, 1], title="probability")
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(style_figure(fig, 320, "Class probabilities"), width="stretch")

st.subheader("Grad-CAM explanation")
st.caption(
    f"Target: **{display_name(exp.target_class)}** (predicted class) · layer `{exp.target_layer}` · "
    "images shown at model input resolution"
)
images = {
    "original": base64.b64decode(exp.original_png_base64),
    "heatmap": base64.b64decode(exp.heatmap_png_base64),
    "overlay": base64.b64decode(exp.overlay_png_base64),
}
cols = st.columns(3)
for col, (label, png) in zip(cols, images.items(), strict=True):
    with col:
        st.image(png, caption=label.capitalize(), width="stretch")
        st.download_button(
            f"Download {label}",
            png,
            file_name=f"neurolens_{label}.png",
            mime="image/png",
            key=f"dl_{label}",
            width="stretch",
        )
st.image(base64.b64decode(exp.legend_png_base64), width=480)
st.caption(exp.legend)
st.info(exp.interpretation_note)
if exp.degenerate:
    st.warning(
        "The Grad-CAM map is all zeros for this image: the target layer had no positive "
        "gradient-weighted evidence. The heatmap is intentionally blank."
    )

report = {
    "prediction": pred.model_dump(),
    "explanation": {k: v for k, v in exp.model_dump().items() if not k.endswith("_base64")},
}
st.download_button(
    "Download JSON report",
    json.dumps(report, indent=2),
    file_name="neurolens_report.json",
    mime="application/json",
    icon=":material/download:",
)
