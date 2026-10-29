"""Documentation: renders the Markdown guides from ``docs/`` so the app and repo never diverge."""

from __future__ import annotations

import streamlit as st

from ui import DOCS_DIR, disclaimer, header

header(
    "Guides", "Documentation", "How NeuroLens works, how to reproduce it, and where it falls short."
)
disclaimer()

SECTIONS = {
    "How the model works": "how_it_works.md",
    "Dataset setup": "setup.md",
    "Training & evaluation": "training_evaluation.md",
    "Grad-CAM": "explainability.md",
    "ONNX export": "onnx.md",
    "Deployment": "deployment.md",
    "API": "api.md",
    "Limitations": "limitations.md",
}
tabs = st.tabs(list(SECTIONS))
for tab, filename in zip(tabs, SECTIONS.values(), strict=True):
    with tab:
        path = DOCS_DIR / filename
        if path.exists():
            st.markdown(path.read_text(encoding="utf-8"))
        else:
            st.warning(f"docs/{filename} is missing from this checkout.")
