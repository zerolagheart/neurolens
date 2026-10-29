"""NeuroLens Streamlit entry point.

streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit puts only this file's directory on sys.path; add the repo root (for `api`) and
# `src` (for an uninstalled checkout) so imports work the same as under uvicorn/pytest.
_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="NeuroLens", page_icon=":material/neurology:", layout="wide")

from ui import apply_theme  # noqa: E402

apply_theme()

PAGES = Path(__file__).parent / "views"
navigation = st.navigation(
    [
        st.Page(PAGES / "overview.py", title="Overview", icon=":material/dashboard:", default=True),
        st.Page(PAGES / "analysis.py", title="MRI Analysis", icon=":material/biotech:"),
        st.Page(PAGES / "laboratory.py", title="Model Laboratory", icon=":material/science:"),
        st.Page(PAGES / "dataset.py", title="Dataset Explorer", icon=":material/dataset:"),
        st.Page(PAGES / "documentation.py", title="Documentation", icon=":material/menu_book:"),
    ]
)
with st.sidebar:
    st.caption("NeuroLens · research & education only · not a medical device")
navigation.run()
