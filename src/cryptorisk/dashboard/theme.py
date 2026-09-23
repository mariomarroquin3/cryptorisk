"""Dark, dense, terminal-style theme shared by every page."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

BG = "#0b0e11"
PANEL = "#12161a"
GRID = "#1e242b"
TEXT = "#d7dce1"
MUTED = "#7b8794"
AMBER = "#ffb020"
GREEN = "#3ddc84"
RED = "#ff4d4f"
BLUE = "#4dabf7"
VIOLET = "#b083f0"

_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT, family="JetBrains Mono, Consolas, monospace", size=12),
        colorway=[AMBER, BLUE, GREEN, RED, VIOLET, "#ff8f40", "#40c4ff"],
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=20, t=40, b=30),
    )
)
pio.templates["cubo_dark"] = _TEMPLATE
pio.templates.default = "cubo_dark"


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {BG}; }}
        [data-testid="stSidebar"] {{ background-color: {PANEL}; border-right: 1px solid {GRID}; }}
        [data-testid="stMetric"] {{
            background-color: {PANEL}; border: 1px solid {GRID}; border-radius: 4px;
            padding: 10px 14px;
        }}
        [data-testid="stMetricLabel"] {{ color: {MUTED}; font-size: 0.75rem; text-transform: uppercase; letter-spacing: .04em; }}
        [data-testid="stMetricValue"] {{ font-family: 'JetBrains Mono', Consolas, monospace; }}
        h1, h2, h3 {{ font-family: 'JetBrains Mono', Consolas, monospace; letter-spacing: .02em; }}
        .cubo-badge {{
            display: inline-block; padding: 2px 8px; border-radius: 3px;
            font-size: 0.7rem; font-weight: 600; letter-spacing: .05em; text-transform: uppercase;
        }}
        .cubo-badge-live {{ background: rgba(61,220,132,.15); color: {GREEN}; border: 1px solid {GREEN}; }}
        .cubo-badge-frozen {{ background: rgba(77,171,247,.15); color: {BLUE}; border: 1px solid {BLUE}; }}
        .cubo-ticker {{ font-family: 'JetBrains Mono', Consolas, monospace; }}
        div[data-testid="stDataFrame"] {{ border: 1px solid {GRID}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_setup(title: str) -> None:
    st.set_page_config(page_title=f"Cryptorisk | {title}", page_icon="◈", layout="wide")
    inject_css()
