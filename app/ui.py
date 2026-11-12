"""Shared look-and-feel: CSS tokens, Plotly theme, small HTML components, cached loaders."""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scam_analyzer.config import MODELS_DIR, REPORTS_DIR  # noqa: E402

# ------------------------------------------------------------------ tokens
C = {
    "bg": "#0A1424", "panel": "#0F1D33", "panel2": "#132541", "line": "#1F3352",
    "text": "#E6EDF7", "text2": "#A3B3CB", "muted": "#6F82A0",
    "green": "#3DF5A0", "cyan": "#38D6F5", "red": "#FF5470", "amber": "#F5B83D",
    # categorical slots for multi-series charts (validated dark-mode order: blue, orange, aqua)
    "s1": "#3987e5", "s2": "#d95926", "s3": "#199e70",
}
PAGES: dict = {}   # filled by streamlit_app.py so pages can link to each other
RISK_COLOR = {"HIGH": C["red"], "MEDIUM": C["amber"], "LOW": C["green"]}
RISK_ICON = {"HIGH": "▲", "MEDIUM": "◆", "LOW": "●"}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Sans+Condensed:wght@500;600;700&display=swap');
:root {{ --bg:{C['bg']}; --panel:{C['panel']}; --panel2:{C['panel2']}; --line:{C['line']}; --text:{C['text']};
        --text2:{C['text2']}; --muted:{C['muted']}; --green:{C['green']}; --cyan:{C['cyan']}; --red:{C['red']}; --amber:{C['amber']}; }}
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select {{
  font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif !important; }}
.stApp {{ background: radial-gradient(1200px 600px at 85% -10%, rgba(56,214,245,.07), transparent 60%), var(--bg); color: var(--text); }}
section[data-testid="stSidebar"] {{ background: #081120; border-right: 1px solid var(--line); }}
h1, h2, h3 {{ color: var(--text); letter-spacing: -0.01em; }}
h1 {{ font-weight: 600; font-size: 2.1rem !important; line-height: 1.15; }}
h2 {{ font-weight: 600; font-size: 1.35rem !important; margin-top: 1.4rem !important; }}
h3 {{ font-weight: 500; font-size: 1.05rem !important; color: var(--text2); }}
p, li {{ line-height: 1.6; }}
.block-container {{ padding-top: 2.2rem; max-width: 1280px; }}
a {{ color: var(--cyan) !important; }}
:focus-visible {{ outline: 2px solid var(--cyan) !important; outline-offset: 2px; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid var(--line); }}
.stTabs [data-baseweb="tab"] {{ background: transparent; color: var(--text2); padding: 8px 14px; }}
.stTabs [aria-selected="true"] {{ color: var(--text) !important; border-bottom: 2px solid var(--cyan) !important; }}
div[data-testid="stMetric"] {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; }}
div[data-testid="stMetricValue"] {{ font-family: 'IBM Plex Sans Condensed', sans-serif !important; font-variant-numeric: tabular-nums; }}
div[data-testid="stMetricLabel"] p {{ color: var(--text2) !important; }}
textarea {{ background: #081120 !important; color: var(--text) !important; border: 1px solid var(--line) !important;
           font-size: 0.95rem !important; line-height: 1.55 !important; }}
.stButton > button {{ border-radius: 8px; border: 1px solid var(--line); background: var(--panel2); color: var(--text); }}
.stButton > button[kind="primary"] {{ background: var(--green); color: #04210f; border: none; font-weight: 600; }}
.stButton > button:hover {{ border-color: var(--cyan); }}
.stDataFrame {{ border: 1px solid var(--line); border-radius: 8px; }}
/* components */
.lede {{ color: var(--text2); font-size: 1.08rem; max-width: 70ch; }}
.panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 18px 20px; margin-bottom: 12px; }}
.panel.thin {{ padding: 12px 16px; }}
.note {{ border-left: 3px solid var(--cyan); background: rgba(56,214,245,.06); padding: 10px 14px; border-radius: 0 8px 8px 0;
        color: var(--text2); font-size: .93rem; margin: 8px 0 14px; }}
.note.warn {{ border-left-color: var(--amber); background: rgba(245,184,61,.07); }}
.note.bad {{ border-left-color: var(--red); background: rgba(255,84,112,.07); }}
.pipeline {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 0; margin: 10px 0 6px; }}
.pipe-step {{ position: relative; padding: 14px 14px 14px 16px; background: var(--panel); border: 1px solid var(--line); border-right: none; }}
.pipe-step:first-child {{ border-radius: 10px 0 0 10px; }}
.pipe-step:last-child {{ border-radius: 0 10px 10px 0; border-right: 1px solid var(--line); }}
.pipe-step b {{ display:block; color: var(--text); font-weight: 600; font-size: .95rem; }}
.pipe-step span {{ color: var(--text2); font-size: .82rem; line-height: 1.4; display:block; margin-top: 4px; }}
.pipe-step i {{ font-style: normal; color: var(--cyan); font-size: .78rem; }}
.pipe-step:not(:last-child)::after {{ content: ''; position: absolute; right: -7px; top: 50%; width: 12px; height: 12px; z-index: 2;
        background: var(--panel); border-top: 1px solid var(--line); border-right: 1px solid var(--line); transform: translateY(-50%) rotate(45deg); }}
@media (max-width: 900px) {{ .pipeline {{ grid-template-columns: 1fr 1fr; }} .pipe-step::after {{ display:none; }}
   .pipe-step, .pipe-step:first-child, .pipe-step:last-child {{ border:1px solid var(--line); border-radius: 8px; margin: 3px; }} }}
.verdict {{ border-radius: 14px; padding: 20px 22px; border: 1px solid var(--line); background: var(--panel); }}
.verdict .label {{ font-family: 'IBM Plex Sans Condensed', sans-serif; font-size: 2.6rem; font-weight: 700; line-height: 1; }}
.verdict .sub {{ color: var(--text2); margin-top: 6px; font-size: .95rem; }}
.meter {{ position: relative; height: 14px; border-radius: 7px; margin: 22px 0 6px;
          background: linear-gradient(90deg, rgba(61,245,160,.35) 0 var(--m), rgba(245,184,61,.35) var(--m) var(--h), rgba(255,84,112,.38) var(--h) 100%); }}
.meter .needle {{ position: absolute; top: -7px; width: 4px; height: 28px; border-radius: 2px; background: var(--text); box-shadow: 0 0 0 2px var(--bg); }}
.meter-scale {{ position: relative; height: 18px; color: var(--muted); font-size: .78rem; }}
.meter-scale span {{ position: absolute; top: 0; white-space: nowrap; }}
.meter-scale span:nth-child(2), .meter-scale span:nth-child(3) {{ transform: translateX(-50%); }}
.big-prob {{ font-family: 'IBM Plex Sans Condensed', sans-serif; font-size: 3.2rem; font-weight: 600; font-variant-numeric: tabular-nums; line-height: 1; }}
.chip {{ display:inline-block; padding: 3px 10px; border-radius: 999px; font-size: .82rem; margin: 2px 4px 2px 0; border: 1px solid var(--line); color: var(--text2); }}
.chip.hi {{ border-color: rgba(255,84,112,.6); color: #ffb3c0; background: rgba(255,84,112,.08); }}
.chip.md {{ border-color: rgba(245,184,61,.6); color: #ffe0a3; background: rgba(245,184,61,.07); }}
.chip.ok {{ border-color: rgba(61,245,160,.55); color: #b8ffde; background: rgba(61,245,160,.07); }}
.ind {{ border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; margin-bottom: 8px; background: var(--panel); }}
.ind .t {{ font-weight: 600; }} .ind .w {{ color: var(--text2); font-size: .9rem; margin: 4px 0 6px; }}
.ind.hi {{ border-left: 4px solid var(--red); }} .ind.md {{ border-left: 4px solid var(--amber); }}
.transcript {{ background: #081120; border: 1px solid var(--line); border-radius: 10px; padding: 16px 18px; line-height: 1.75;
              white-space: pre-wrap; max-height: 420px; overflow-y: auto; font-size: .95rem; color: var(--text); }}
mark.r {{ background: rgba(255,84,112,.22); color: #ffd0d8; border-bottom: 2px solid var(--red); padding: 0 2px; border-radius: 3px; }}
mark.m {{ background: rgba(56,214,245,.13); color: #c9f4ff; border-bottom: 1px dashed var(--cyan); padding: 0 2px; border-radius: 3px; }}
mark.g {{ background: rgba(61,245,160,.12); color: #c8ffe6; border-bottom: 1px dashed var(--green); padding: 0 2px; border-radius: 3px; }}
.legend-row {{ color: var(--text2); font-size: .85rem; margin: 6px 0 4px; }}
.kv {{ display:grid; grid-template-columns: 1fr auto; gap: 4px 16px; font-size: .92rem; }}
.kv div:nth-child(odd) {{ color: var(--text2); }} .kv div:nth-child(even) {{ text-align:right; font-variant-numeric: tabular-nums; }}
.foot {{ color: var(--muted); font-size: .82rem; margin-top: 30px; border-top: 1px solid var(--line); padding-top: 10px; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ plotly theme
_tpl = go.layout.Template()
_tpl.layout = go.Layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans, system-ui, sans-serif", color=C["text2"], size=13),
    title=dict(font=dict(color=C["text"], size=15), x=0, xanchor="left"),
    xaxis=dict(gridcolor="rgba(163,179,203,.10)", zerolinecolor="rgba(163,179,203,.25)", linecolor=C["line"], tickcolor=C["line"]),
    yaxis=dict(gridcolor="rgba(163,179,203,.10)", zerolinecolor="rgba(163,179,203,.25)", linecolor=C["line"], tickcolor=C["line"]),
    colorway=[C["s1"], C["s2"], C["s3"]],
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["text2"]), orientation="h", y=1.08, x=0),
    margin=dict(l=10, r=10, t=40, b=10),
    hoverlabel=dict(bgcolor=C["panel2"], bordercolor=C["line"], font=dict(color=C["text"], family="IBM Plex Sans")),
)
pio.templates["scamviz"] = _tpl
pio.templates.default = "scamviz"
SEQ_SCALE = [[0, "#0F1D33"], [0.5, "#256abf"], [1, "#86b6ef"]]          # one hue, dark → light on dark surface
DIV_SCALE = [[0, C["s2"]], [0.5, "#2A3B57"], [1, C["s1"]]]              # orange ↔ neutral ↔ blue


def plot(fig, height=None, key=None):
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False}, key=key)


# ------------------------------------------------------------------ loaders
@st.cache_data(show_spinner=False)
def load_json(name: str):
    p = REPORTS_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_resource(show_spinner="Loading models …")
def get_analyzer():
    from scam_analyzer.risk import ScamAnalyzer
    return ScamAnalyzer()


@st.cache_resource(show_spinner=False)
def get_hmm_bundle():
    import joblib
    p = MODELS_DIR / "hmm_models.joblib"
    return joblib.load(p) if p.exists() else None


def models_ready() -> bool:
    return (MODELS_DIR / "tfidf_vectorizer.joblib").exists() and (MODELS_DIR / "meta.joblib").exists()


def missing_models_message():
    st.error("Trained models were not found in `models/`. Train them first:")
    st.code("python scripts/train_models.py        # ≈ 15–25 min on a laptop\npython scripts/run_analysis.py        # EDA + clustering/HMM reports", language="bash")


# ------------------------------------------------------------------ html helpers
def esc(s) -> str:
    return html.escape(str(s))


def note(text: str, kind: str = "") -> None:
    st.markdown(f'<div class="note {kind}">{text}</div>', unsafe_allow_html=True)


def pct(x, d=1) -> str:
    return "—" if x is None else f"{x * 100:.{d}f}%"


def link(target, page_key: str, label: str, icon: str | None = None) -> None:
    """st.page_link when the page registry exists (real app); plain bold text otherwise (tests)."""
    page = PAGES.get(page_key)
    if page is not None:
        target.page_link(page, label=label, icon=icon)
    else:
        target.markdown(f"**{label}**")


def footer() -> None:
    st.markdown('<div class="foot">AI Scam Conversation Risk Analyzer. Results are model-assisted assessments, not fraud determinations. '
                'Runs fully offline once the models are trained.</div>', unsafe_allow_html=True)
