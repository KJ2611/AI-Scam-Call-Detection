"""Page 1 — Home dashboard."""
from __future__ import annotations

import streamlit as st

import ui
from ui import esc, load_json, models_ready, note, pct


PIPE = [
    ("Transcript", "Pasted call or chat text"),
    ("Preprocessing", "Lower-case; URL, phone, money, number placeholders; source names masked"),
    ("TF-IDF", "10,000 uni- and bigram features, fitted on training data only"),
    ("Classification", "Calibrated linear model (or any of 8 models)"),
    ("Risk assessment", "Probability → Low / Medium / High; rule layer can escalate"),
    ("Explanation", "Exact word contributions + red-flag phrases + advice"),
]


def render():
    m = load_json("metrics.json")
    st.title("AI Scam Conversation Risk Analyzer")
    st.markdown('<p class="lede">Post-call scam screening for conversation transcripts. Paste what was said; get a calibrated scam probability, '
                'the words that drove it, the manipulation tactics present, and what to do next.</p>', unsafe_allow_html=True)
    c1, c2, _ = st.columns([1.2, 1.2, 3])
    ui.link(c1, "detect", "Analyze a transcript", "🛡️")
    ui.link(c2, "models", "See model results", "📊")

    if not m or not models_ready():
        note("Models have not been trained yet. Run <code>python scripts/train_models.py</code>, then reload.", "warn")
        return
    best = m["models"][m["best_model"]]
    d = m["data"]["splits"]
    st.markdown("### How it works")
    st.markdown('<div class="pipeline">' + "".join(
        f'<div class="pipe-step"><i>Step {i}</i><b>{esc(t)}</b><span>{esc(s)}</span></div>' for i, (t, s) in enumerate(PIPE, 1)) + "</div>",
        unsafe_allow_html=True)

    st.markdown("### At a glance")
    k = st.columns(4)
    k[0].metric("Unique conversations used", f"{d['train_full']['rows'] + d['main_test']['rows'] + d['ext_test']['rows']:,}",
                help="Composite (cleaned, de-duplicated) + English scripts")
    k[1].metric("Trained models", len(m["models"]))
    k[2].metric("Default model", best["info"]["name"].split(" (")[0], help=f"{best['info']['name']} — highest mean 5-fold CV F1 on the training split")
    k[3].metric("CV F1 (training, 5-fold)", pct(best["cv"]["f1"]["mean"], 2), help=f"Standard deviation across folds: ± {best['cv']['f1']['std']*100:.2f} points")
    k = st.columns(4)
    mt, et = best["test"]["main_test"], best["test"]["ext_test"]
    k[0].metric("In-distribution test F1", pct(mt["f1"], 2), help=f"n = {mt['n']:,}")
    k[1].metric("In-distribution recall", pct(mt["recall"], 2), help="Share of scams caught")
    k[2].metric("External test accuracy", pct(et["accuracy"], 1), help=f"Held-out English scripts, n = {et['n']}")
    k[3].metric("Out-of-domain accuracy", pct(m["ablations"]["out_of_domain_english"][m["best_model"]]["accuracy"], 1)
                if m["best_model"] in m["ablations"]["out_of_domain_english"] else "—",
                help="Trained on composite only, tested on all English scripts")
    note("Read the last two numbers together with the first two. Near-perfect in-distribution scores partly reflect how the composite dataset was "
         "assembled (scam and non-scam rows come from different sources). Accuracy on text from a different source is lower — see Model comparison.",
         "warn")

    st.markdown("### Sections")
    a, b, c = st.columns(3)
    with a:
        ui.link(st, "detect", "Scam detection", "🛡️")
        st.caption("Live analysis with evidence, red flags, advice and statistics.")
        ui.link(st, "models", "Model comparison", "📊")
        st.caption("Eight models, four test sets, calibration and ablations.")
    with b:
        ui.link(st, "dataset", "Dataset explorer", "🗂️")
        st.caption("Sources, cleaning, lengths, vocabulary and corrected EDA.")
        ui.link(st, "advanced", "Advanced ML", "🧬")
        st.caption("Clustering and Hidden Markov Models (supplementary).")
    with c:
        ui.link(st, "about", "About the project", "📘")
        st.caption("Problem, method, stack, limitations and references.")
