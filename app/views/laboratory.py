"""Model Laboratory: measured evaluation results, training curves, calibration and benchmarks."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from neurolens.data.classes import display_name
from neurolens.evaluation.compare import collect_runs
from ui import (
    ACCENT,
    INK,
    RUNS_DIR,
    SEQUENTIAL,
    SERIES,
    disclaimer,
    empty_state,
    fmt,
    header,
    load_json,
    style_figure,
)

header(
    "Offline evaluation",
    "Model Laboratory",
    "Measured results from saved evaluation files. Experiments that have not been run are "
    "shown as unavailable rather than estimated.",
)
disclaimer()

rows = collect_runs(RUNS_DIR)
table = pd.DataFrame(rows)

st.subheader("Architecture comparison")
show_cols = [
    c
    for c in [
        "architecture",
        "run",
        "status",
        "benchmark",
        "test_n",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "roc_auc_ovr_macro",
        "ece_raw",
        "ece_calibrated",
        "parameters",
        "checkpoint_mb",
        "latency_device",
        "latency_b1_ms",
        "throughput_b32_img_s",
        "onnx_b1_ms",
    ]
    if c in table.columns
]
st.dataframe(
    table[show_cols],
    hide_index=True,
    width="stretch",
    column_config={
        c: st.column_config.NumberColumn(format="%.4f")
        for c in [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "roc_auc_ovr_macro",
            "ece_raw",
            "ece_calibrated",
        ]
        if c in show_cols
    },
)
if "benchmark" in table and table["benchmark"].dropna().nunique() > 1:
    st.warning(
        "Runs use different benchmarks. Metrics on the leakage-audited test subset are not "
        "comparable with the original test set."
    )
if "latency_device" in table and table["latency_device"].dropna().nunique() > 1:
    st.warning("Latency was measured on different devices; compare latency only within one device.")

trained = [r for r in rows if r["run"]]
if not trained:
    empty_state(
        "No trained runs yet",
        "Train and evaluate a model to populate this page:",
        "neurolens train --config configs/resnet18.yaml\nneurolens calibrate --run resnet18_baseline\n"
        "neurolens evaluate --run resnet18_baseline\nneurolens compare",
    )
    st.stop()

st.divider()
run = st.selectbox("Inspect run", [r["run"] for r in trained])
run_dir = RUNS_DIR / run
metrics = load_json(run_dir / "eval" / "test" / "metrics.json")
history_path = run_dir / "history.csv"

tab_eval, tab_curves, tab_cal, tab_perf = st.tabs(
    ["Test metrics", "Training curves", "Calibration & abstention", "Latency & ONNX"]
)

with tab_eval:
    if not metrics:
        empty_state(
            "Not evaluated",
            "This run has no test evaluation yet.",
            f"neurolens calibrate --run {run}\nneurolens evaluate --run {run}",
        )
    else:
        m = metrics["metrics"]
        st.caption(
            f"Benchmark `{metrics['benchmark']}` · split `{metrics['split']}` · n = {m['n_samples']} · "
            f"checkpoint epoch {metrics['checkpoint_epoch']} · evaluated {metrics['evaluated_at']}"
        )
        if metrics.get("note"):
            st.info(metrics["note"])
        c = st.columns(5)
        c[0].metric("Accuracy", fmt(m["accuracy"], 4))
        c[1].metric("Balanced acc.", fmt(m["balanced_accuracy"], 4))
        c[2].metric("Macro F1", fmt(m["macro_f1"], 4))
        c[3].metric("ROC-AUC (OvR macro)", fmt(m["roc_auc"]["macro"], 4))
        c[4].metric("Log loss", fmt(m["log_loss"], 4))
        names = [display_name(n) for n in metrics["class_names"]]
        left, right = st.columns(2)
        with left:
            cm = m["confusion_matrix"]
            totals = [max(sum(r), 1) for r in cm]
            norm = [[v / t for v in r] for r, t in zip(cm, totals, strict=True)]
            fig = go.Figure(
                go.Heatmap(
                    z=norm,
                    x=names,
                    y=names,
                    colorscale=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)],
                    zmin=0,
                    zmax=1,
                    text=cm,
                    texttemplate="%{text}",
                    textfont=dict(color=INK),
                    hovertemplate="true %{y}<br>predicted %{x}<br>count %{text}<br>row share %{z:.3f}<extra></extra>",
                    colorbar=dict(title="row share"),
                )
            )
            fig.update_yaxes(autorange="reversed", title="true class")
            fig.update_xaxes(title="predicted class")
            st.plotly_chart(
                style_figure(fig, 380, "Confusion matrix (counts, coloured by row share)"),
                width="stretch",
            )
        with right:
            fig = go.Figure()
            for k, metric in enumerate(["precision", "recall", "f1"]):
                fig.add_bar(
                    x=names,
                    y=[m["per_class"][n][metric] for n in metrics["class_names"]],
                    name=metric,
                    marker_color=SERIES[k],
                    hovertemplate="%{x}<br>" + metric + ": %{y:.4f}<extra></extra>",
                )
            fig.update_yaxes(range=[0, 1.02])
            st.plotly_chart(
                style_figure(fig, 380, "Per-class precision / recall / F1"), width="stretch"
            )
        per_class = pd.DataFrame(m["per_class"]).T.rename(index=display_name)
        st.dataframe(per_class, width="stretch")
        with st.expander("Classification report"):
            st.code(m["classification_report"])

with tab_curves:
    if not history_path.exists():
        st.info("No training history found for this run.")
    else:
        history = pd.read_csv(history_path)
        left, right = st.columns(2)
        for col, metric in ((left, "loss"), (right, "accuracy")):
            fig = go.Figure()
            fig.add_scatter(
                x=history["epoch"],
                y=history[f"train_{metric}"],
                mode="lines+markers",
                name="train",
                line=dict(color=SERIES[0], width=2),
                marker=dict(size=8),
            )
            fig.add_scatter(
                x=history["epoch"],
                y=history[f"val_{metric}"],
                mode="lines+markers",
                name="validation",
                line=dict(color=SERIES[1], width=2),
                marker=dict(size=8),
            )
            fig.update_xaxes(title="epoch", dtick=1)
            fig.update_layout(hovermode="x unified")
            with col:
                st.plotly_chart(
                    style_figure(fig, 320, f"{metric.capitalize()} per epoch"), width="stretch"
                )
        st.caption(
            "Training metrics are computed on augmented batches in train mode; validation metrics "
            "on the un-augmented validation split. The best checkpoint is selected on validation only."
        )
        st.dataframe(history, hide_index=True, width="stretch")

with tab_cal:
    calibration = load_json(run_dir / "calibration.json")
    if not metrics:
        st.info("Run `neurolens evaluate` to see test-set calibration.")
    else:
        details = metrics["calibration"]["details"]
        c = st.columns(3)
        c[0].metric(
            "Temperature T",
            fmt(metrics["calibration"]["temperature"], 3),
            help="fitted on the validation split" if calibration else "not fitted",
        )
        c[1].metric(f"ECE raw ({details['ece_bins']} bins)", fmt(details["raw"]["ece"], 4))
        c[2].metric("ECE calibrated", fmt(details.get("calibrated", {}).get("ece"), 4))
        left, right = st.columns(2)
        with left:
            fig = go.Figure()
            fig.add_scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="perfect calibration",
                line=dict(color="#64748b", dash="dash", width=1.5),
                hoverinfo="skip",
            )
            for k, key in enumerate(["raw", "calibrated"]):
                if key in details:
                    bins = [b for b in details[key]["reliability"] if b["count"]]
                    fig.add_scatter(
                        x=[b["confidence"] for b in bins],
                        y=[b["accuracy"] for b in bins],
                        mode="lines+markers",
                        name=key,
                        line=dict(color=SERIES[k], width=2),
                        marker=dict(size=8),
                        customdata=[b["count"] for b in bins],
                        hovertemplate="confidence %{x:.3f}<br>accuracy %{y:.3f}<br>n=%{customdata}<extra>"
                        + key
                        + "</extra>",
                    )
            fig.update_xaxes(range=[0, 1], title="mean confidence in bin")
            fig.update_yaxes(range=[0, 1], title="accuracy in bin")
            st.plotly_chart(
                style_figure(fig, 360, "Reliability diagram (test split)"), width="stretch"
            )
        with right:
            fig = go.Figure()
            for k, key in enumerate(["raw", "calibrated"]):
                if key in details:
                    cr = details[key]["coverage_risk"]
                    fig.add_scatter(
                        x=[r["coverage"] for r in cr],
                        y=[r["risk"] for r in cr],
                        mode="lines+markers",
                        name=key,
                        line=dict(color=SERIES[k], width=2),
                        marker=dict(size=8),
                        customdata=[r["threshold"] for r in cr],
                        hovertemplate="threshold %{customdata}<br>coverage %{x:.3f}<br>error rate %{y:.4f}<extra>"
                        + key
                        + "</extra>",
                    )
            fig.update_xaxes(title="coverage (fraction accepted)")
            fig.update_yaxes(title="error rate among accepted")
            st.plotly_chart(
                style_figure(fig, 360, "Coverage vs. risk when abstaining"), width="stretch"
            )
        st.caption(
            "Each point is a confidence threshold: images below it are abstained on. Thresholds "
            "are illustrative configuration, not clinically validated safety thresholds."
        )

with tab_perf:
    bench_files = sorted((run_dir / "benchmarks").glob("pytorch_*.json"))
    onnx = load_json(run_dir / "onnx" / "onnx_report.json")
    if not bench_files and not onnx:
        empty_state(
            "No performance measurements",
            "Latency has not been measured for this run.",
            f"neurolens benchmark --run {run}\nneurolens export-onnx --run {run} --quantize",
        )
    for path in bench_files:
        bench = load_json(path)
        st.markdown(f"**PyTorch · {bench['device']}** — {bench['procedure']}")
        frame = pd.DataFrame(bench["latency"])
        st.dataframe(frame, hide_index=True, width="stretch")
        extra = [f"parameters: {bench['parameters']['total']:,}"]
        if "checkpoint_size_mb" in bench:
            extra.append(f"checkpoint: {bench['checkpoint_size_mb']:.1f} MB")
        if "preprocess_ms_single_image" in bench:
            extra.append(
                f"decode + preprocess (1 image): {bench['preprocess_ms_single_image']:.2f} ms"
            )
        st.caption(" · ".join(extra) + f" · measured {bench['measured_at']}")
    if onnx:
        st.markdown(
            f"**ONNX Runtime (CPU)** — opset {onnx['opset']}, parity on {onnx['parity_inputs']}"
        )
        p = onnx["parity"]
        c = st.columns(4)
        c[0].metric("ONNX size", f"{onnx['onnx']['size_mb']:.1f} MB")
        c[1].metric("max |Δp|", f"{p['max_abs_prob_diff']:.2e}")
        c[2].metric(
            "Parity",
            "PASS" if p["within_tolerance"] else "FAIL",
            help=f"tolerance {p['tolerance']:g}",
        )
        c[3].metric("Top-1 agreement", f"{p['top1_agreement']:.3f}")
        rows = []
        for key, label in (("pytorch_cpu", "PyTorch CPU (fp32)"), ("onnx", "ONNX Runtime (fp32)")):
            for r in onnx["benchmark"][key]:
                rows.append({"runtime": label, **r})
        q = onnx.get("quantized")
        if q and not q.get("error"):
            for r in q["benchmark"]:
                rows.append({"runtime": "ONNX Runtime (int8 dynamic)", **r})
        frame = pd.DataFrame(rows)
        fig = go.Figure()
        for k, (runtime, group) in enumerate(frame.groupby("runtime", sort=False)):
            fig.add_bar(
                x=[f"batch {b}" for b in group["batch_size"]],
                y=group["mean_ms"],
                name=runtime,
                marker_color=SERIES[k],
                hovertemplate="%{x}<br>%{y:.2f} ms<extra>" + runtime + "</extra>",
            )
        fig.update_yaxes(title="mean latency (ms)")
        st.plotly_chart(
            style_figure(fig, 320, "CPU latency by runtime (same machine, same inputs)"),
            width="stretch",
        )
        st.dataframe(frame, hide_index=True, width="stretch")
        if q:
            if q.get("error"):
                st.warning(f"Dynamic quantization failed: {q['error']}")
            else:
                st.caption(
                    f"int8 dynamic: {q['size_mb']:.1f} MB, max |Δp| {q['parity']['max_abs_prob_diff']:.2e}, "
                    f"top-1 agreement {q['parity']['top1_agreement']:.3f} vs. PyTorch. Reported as measured, "
                    "including any regression."
                )
        st.caption(
            f"Conditions: {onnx['benchmark_conditions']['device']} · {onnx['benchmark_conditions']['procedure']}"
        )

st.markdown(
    f'<span style="color:{ACCENT}">●</span> All values on this page are read from files produced by the '
    "CLI; nothing is computed from live traffic or estimated.",
    unsafe_allow_html=True,
)
