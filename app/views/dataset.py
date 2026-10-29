"""Dataset Explorer: audit results, distributions, duplicates and limitations."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from neurolens.data.classes import CANONICAL_CLASSES, display_name
from ui import (
    AUDIT_DIR,
    DATA_ROOT,
    SERIES,
    SPLITS_DIR,
    disclaimer,
    empty_state,
    header,
    load_json,
    stat,
    style_figure,
)

header(
    "Data integrity",
    "Dataset Explorer",
    "Class and split distributions, image properties, duplicate findings and known limitations "
    "of the Nickparvar Brain Tumor MRI dataset.",
)
disclaimer()

report = load_json(AUDIT_DIR / "audit_report.json")
if not report:
    empty_state(
        "No audit report yet",
        "Download the dataset from Kaggle and run the audit. It only reads files; nothing is "
        "moved or deleted.",
        "neurolens audit --data-root data/raw/brain-tumor-mri-dataset",
    )
    st.stop()

exact, near, leak = report["exact_duplicates"], report["near_duplicates"], report["leakage"]
c = st.columns(4)
with c[0]:
    stat("Images", f"{report['total_files']:,}", f"{len(report['unreadable_files'])} unreadable")
with c[1]:
    stat(
        "Exact duplicate pairs",
        f"{exact['pairs']:,}",
        f"{exact['cross_partition_pairs']} across partitions",
    )
with c[2]:
    stat(
        "Near-duplicate candidates",
        f"{near['pairs']:,}",
        f"pHash Hamming ≤ {near['threshold']} · {near['cross_partition_pairs']:,} cross-partition",
    )
with c[3]:
    frac = leak.get("test_fraction_with_match_in_train")
    stat(
        "Test images linked to train",
        f"{leak.get('test_files_with_match_in_train', 0):,}",
        f"{frac * 100:.1f}% of test" if frac is not None else "",
    )
st.caption(
    f"Audit generated {report['generated_at']} · paths are shown relative to the dataset root."
)

st.subheader("Class distribution by partition")
counts = report["counts_by_partition_and_class"]
fig = go.Figure()
for k, cls in enumerate(CANONICAL_CLASSES):
    fig.add_bar(
        x=list(counts),
        y=[counts[p].get(cls, 0) for p in counts],
        name=display_name(cls),
        marker_color=SERIES[k],
        hovertemplate="%{x}<br>" + display_name(cls) + ": %{y}<extra></extra>",
    )
fig.update_layout(barmode="group")
fig.update_yaxes(title="images")
st.plotly_chart(style_figure(fig, 320), width="stretch")

st.subheader("Split manifests")
manifests = []
for meta_path in sorted(SPLITS_DIR.glob("*/manifest_meta.json")):
    meta = load_json(meta_path)
    for split, per_class in meta["counts"].items():
        manifests.append(
            {
                "manifest": meta_path.parent.name,
                "benchmark": meta["benchmark"],
                "split": split,
                "total": sum(per_class.values()),
                **{display_name(k): v for k, v in per_class.items()},
                "excluded files": len(meta["excluded"]),
            }
        )
if manifests:
    st.dataframe(pd.DataFrame(manifests), hide_index=True, width="stretch")
    st.caption(
        "`original` keeps the dataset's Testing folder intact. `leakage_audited` is a modified benchmark "
        "(duplicate-linked test images removed, group-aware validation split); its results are not "
        "comparable with the original benchmark."
    )
else:
    st.info("No split manifests yet: `neurolens make-split --config configs/resnet18.yaml`.")

st.subheader("Image properties")
left, mid, right = st.columns([2, 1, 1])
with left:
    sizes = report["image_size"]["most_common"]
    fig = go.Figure(
        go.Bar(
            x=[s["count"] for s in sizes][::-1],
            y=[s["size"] for s in sizes][::-1],
            orientation="h",
            marker_color=SERIES[0],
            hovertemplate="%{y}: %{x} images<extra></extra>",
        )
    )
    fig.update_xaxes(title="images")
    st.plotly_chart(
        style_figure(
            fig, 320, f"Most common sizes ({report['image_size']['unique_sizes']} unique)"
        ),
        width="stretch",
    )
with mid:
    st.markdown("**Formats**")
    st.dataframe(pd.Series(report["formats"], name="files"), width="stretch")
    st.markdown("**Colour modes**")
    st.dataframe(pd.Series(report["color_modes"], name="files"), width="stretch")
with right:
    w, h = report["image_size"]["width"], report["image_size"]["height"]
    st.markdown("**Width / height (px)**")
    st.dataframe(pd.DataFrame({"width": w, "height": h}), width="stretch")
    st.caption("All images are resized to the model input size; aspect ratio is not preserved.")

st.subheader("Duplicates and leakage")
summary = pd.DataFrame(
    [
        {
            "type": "exact (SHA-256)",
            **exact["by_partition_pair"],
            "label conflicts": exact["label_conflict_pairs"],
        },
        {
            "type": f"near (pHash ≤ {near['threshold']})",
            **near["by_partition_pair"],
            "label conflicts": near["label_conflict_pairs"],
        },
    ]
).fillna(0)
st.dataframe(summary, hide_index=True, width="stretch")
st.caption(
    "Columns are partition pairs (e.g. `test-train` = one image in each). Label conflicts are pairs whose "
    "two images carry different class labels."
)

pairs_path = AUDIT_DIR / "duplicate_pairs.csv"
if pairs_path.exists():
    pairs = pd.read_csv(pairs_path)
    kind = st.radio("Show pairs", ["cross-partition", "label conflicts", "all"], horizontal=True)
    if kind == "cross-partition":
        pairs = pairs[pairs["cross_partition"]]
    elif kind == "label conflicts":
        pairs = pairs[pairs["label_conflict"]]
    pairs = pairs.sort_values(["distance", "relpath_a"])
    st.dataframe(pairs.head(500), hide_index=True, width="stretch")
    st.caption(
        f"{len(pairs):,} pairs (first 500 shown). Full list: `artifacts/audit/duplicate_pairs.csv`."
    )
    if DATA_ROOT.exists() and len(pairs):
        with st.expander("Visual check of candidate pairs"):
            st.caption(
                "Near-duplicate candidates need human review: some are re-encoded copies, others are "
                "adjacent slices or unrelated scans with similar global structure."
            )
            sample = pairs.sample(min(6, len(pairs)), random_state=0)
            for _, row in sample.iterrows():
                a, b = DATA_ROOT / row["relpath_a"], DATA_ROOT / row["relpath_b"]
                if not (a.exists() and b.exists()):
                    continue
                cols = st.columns([1, 1, 2])
                cols[0].image(Image.open(a), width="stretch")
                cols[1].image(Image.open(b), width="stretch")
                cols[2].markdown(
                    f"**{row['kind']}**, distance {row['distance']}  \n"
                    f"`{row['relpath_a']}` ({row['partition_a']})  \n"
                    f"`{row['relpath_b']}` ({row['partition_b']})"
                )

st.subheader("Limitations")
for item in report["limitations"]:
    st.markdown(f"- {item}")
if report.get("unreadable_files"):
    with st.expander("Unreadable files"):
        st.dataframe(pd.DataFrame(report["unreadable_files"]), hide_index=True)
