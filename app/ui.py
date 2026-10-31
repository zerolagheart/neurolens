"""Shared UI helpers: theme, layout components, Plotly styling and artifact loading."""

from __future__ import annotations

import os
from html import escape
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from neurolens import DISCLAIMER
from neurolens.utils import read_json

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = Path(os.environ.get("NEUROLENS_RUNS_DIR", ROOT / "artifacts" / "runs"))
AUDIT_DIR = Path(os.environ.get("NEUROLENS_AUDIT_DIR", ROOT / "artifacts" / "audit"))
SPLITS_DIR = Path(os.environ.get("NEUROLENS_SPLITS_DIR", ROOT / "artifacts" / "splits"))
DATA_ROOT = Path(
    os.environ.get("NEUROLENS_DATA_ROOT", ROOT / "data" / "raw" / "brain-tumor-mri-dataset")
)
DOCS_DIR = ROOT / "docs"
API_URL = os.environ.get("NEUROLENS_API_URL", "").rstrip("/")
API_DOCS_URL = (API_URL or "http://localhost:8000") + "/docs"

# Chart palette: categorical dark-mode slots validated against the panel surface (#111a2b).
SURFACE = "#111a2b"
INK = "#e2e8f0"
INK_2 = "#94a3b8"
GRID = "#1f2a3d"
ACCENT = "#2dd4bf"
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500"]
# Capped at a mid blue so light cell labels stay legible on every step.
SEQUENTIAL = ["#16233a", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6"]

CSS = """
<style>
:root { --nl-panel:#111a2b; --nl-border:#1f2a3d; --nl-ink:#e2e8f0; --nl-ink2:#94a3b8;
        --nl-accent:#2dd4bf; --nl-warn:#fab219; }
.block-container { padding-top: 2.2rem; max-width: 1200px; }
h1, h2, h3 { letter-spacing: -0.01em; }
.nl-eyebrow { color: var(--nl-accent); font-size: .78rem; font-weight: 600;
              letter-spacing: .12em; text-transform: uppercase; margin-bottom: .2rem; }
.nl-sub { color: var(--nl-ink2); font-size: 1.02rem; margin-top: -.4rem; max-width: 760px; }
.nl-card { background: var(--nl-panel); border: 1px solid var(--nl-border); border-radius: 10px;
           padding: 1rem 1.1rem; height: 100%; }
.nl-card h4 { margin: 0 0 .35rem 0; font-size: .95rem; color: var(--nl-ink); }
.nl-card p { margin: 0; color: var(--nl-ink2); font-size: .88rem; line-height: 1.45; }
.nl-stat-label { color: var(--nl-ink2); font-size: .78rem; text-transform: uppercase;
                 letter-spacing: .06em; }
.nl-stat-value { color: var(--nl-ink); font-size: 1.6rem; font-weight: 600; line-height: 1.25; }
.nl-stat-note { color: var(--nl-ink2); font-size: .8rem; }
.nl-disclaimer { border: 1px solid #5a4a26; background: #17140d; border-radius: 8px;
                 padding: .7rem 1rem; color: #e9dcc0; font-size: .88rem; margin: .4rem 0 1.2rem 0; }
.nl-disclaimer b { color: var(--nl-warn); }
.nl-empty { border: 1px dashed var(--nl-border); border-radius: 10px; padding: 1.4rem 1.4rem;
            background: #0e1626; }
.nl-empty h4 { margin-top: 0; }
.nl-pill { display:inline-block; padding: .1rem .55rem; border-radius: 999px; font-size: .75rem;
           border: 1px solid var(--nl-border); color: var(--nl-ink2); margin-right: .3rem; }
.nl-pill.ok { color: var(--nl-accent); border-color: #1d4b48; }
.nl-pill.no { color: #c3a565; border-color: #3b3220; }
.nl-pred { font-size: 1.9rem; font-weight: 650; color: var(--nl-ink); line-height: 1.2; }
div[data-testid="stMetricValue"] { font-size: 1.45rem; }
</style>
"""


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header(eyebrow: str, title: str, subtitle: str | None = None) -> None:
    st.markdown(f'<div class="nl-eyebrow">{escape(eyebrow)}</div>', unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.markdown(f'<p class="nl-sub">{escape(subtitle)}</p>', unsafe_allow_html=True)


def disclaimer() -> None:
    lead, _, rest = DISCLAIMER.partition(". ")
    st.markdown(
        f'<div class="nl-disclaimer"><b>{escape(lead)}.</b> {escape(rest)}</div>',
        unsafe_allow_html=True,
    )


def card(title: str, body: str) -> None:
    st.markdown(
        f'<div class="nl-card"><h4>{escape(title)}</h4><p>{escape(body)}</p></div>',
        unsafe_allow_html=True,
    )


def stat(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f'<div class="nl-card"><div class="nl-stat-label">{escape(label)}</div>'
        f'<div class="nl-stat-value">{escape(value)}</div>'
        f'<div class="nl-stat-note">{escape(note)}</div></div>',
        unsafe_allow_html=True,
    )


def empty_state(title: str, body_markdown: str, commands: str | None = None) -> None:
    st.markdown(f'<div class="nl-empty"><h4>{escape(title)}</h4></div>', unsafe_allow_html=True)
    st.markdown(body_markdown)
    if commands:
        st.code(commands, language="bash")


def pill(text: str, ok: bool) -> str:
    return f'<span class="nl-pill {"ok" if ok else "no"}">{escape(text)}</span>'


def style_figure(fig: go.Figure, height: int = 340, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        title=dict(
            text=title, font=dict(size=14, color=INK), x=0, xanchor="left", y=0.98, yanchor="top"
        )
        if title
        else None,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK_2, size=12),
        # Title on the first row, legend on its own row beneath it, so they never collide.
        margin=dict(l=48, r=16, t=80 if title else 40, b=40),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=INK_2)
        ),
        hoverlabel=dict(bgcolor="#0b1220", bordercolor=GRID, font=dict(color=INK)),
        bargap=0.3,
        bargroupgap=0.06,  # small surface gap between adjacent grouped bars
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID, zeroline=False)
    return fig


def load_json(path: Path) -> dict[str, Any] | None:
    return read_json(path) if Path(path).is_file() else None


def fmt(value: Any, digits: int = 3, pct: bool = False) -> str:
    if value is None:
        return "—"
    if pct:
        return f"{value * 100:.1f}%"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)
