"""Tiny harness so Streamlit's AppTest can render one page function at a time."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "app", ROOT / "src"):
    sys.path.insert(0, str(p))

import streamlit as st  # noqa: E402

import ui  # noqa: E402
from views import about, advanced, dataset, detect, home, models  # noqa: E402

ui.inject_css()
page = st.query_params.get("page", "home")
{"home": home, "detect": detect, "models": models, "dataset": dataset, "advanced": advanced, "about": about}[page].render()
