"""AI Scam Conversation Risk Analyzer — Streamlit entry point.

Run from the project root:   streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
for p in (APP_DIR, APP_DIR.parent / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="AI Scam Conversation Risk Analyzer", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")

import ui  # noqa: E402
from views import about, advanced, dataset, detect, home, models  # noqa: E402

ui.inject_css()

pages = [
    st.Page(home.render, title="Home", icon="🏠", url_path="home", default=True),
    st.Page(detect.render, title="Scam detection", icon="🛡️", url_path="detect"),
    st.Page(models.render, title="Model comparison", icon="📊", url_path="models"),
    st.Page(dataset.render, title="Dataset explorer", icon="🗂️", url_path="dataset"),
    st.Page(advanced.render, title="Advanced ML", icon="🧬", url_path="advanced"),
    st.Page(about.render, title="About", icon="📘", url_path="about"),
]
ui.PAGES.update({p.url_path: p for p in pages})
nav = st.navigation(pages)
with st.sidebar:
    st.markdown("**AI Scam Conversation Risk Analyzer**")
    st.caption("Post-call scam screening with NLP and machine learning. Runs locally; no transcript leaves this computer.")
nav.run()
ui.footer()
