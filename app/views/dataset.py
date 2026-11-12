"""Page 4 — Dataset explorer (reports/eda.json)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import C, DIV_SCALE, esc, load_json, note, pct, plot


def _bar_terms(items, title, color, key):
    items = items[:15][::-1]
    fig = go.Figure(go.Bar(x=[v for _, v, *_ in items], y=[t for t, *_ in items], orientation="h",
                           marker=dict(color=color), hovertemplate="%{y}: %{x:,}<extra></extra>"))
    fig.update_layout(title=title, yaxis=dict(automargin=True), bargap=0.35)
    plot(fig, height=430, key=key)


def render():
    st.title("Dataset explorer")
    e = load_json("eda.json")
    m = load_json("metrics.json") or {}
    if not e:
        st.error("reports/eda.json not found — run `python scripts/run_analysis.py`.")
        return
    c = e["counts"]
    st.markdown('<p class="lede">The primary dataset is <code>composite_train.csv</code> (a mix of translated scam dialogues, call transcripts and '
                'SMS-style messages). Four other sources in the project zip are used only for training augmentation or as independent test sets.</p>',
                unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Composite conversations", f"{c['composite_clean_rows']:,}", help="After cleaning and de-duplication")
    k2.metric(f"Scam ({pct(c['scam_share'])})", f"{c['scam']:,}")
    k3.metric(f"Non-scam ({pct(1 - c['scam_share'])})", f"{c['non_scam']:,}")
    k4.metric(f"English scripts ({c['english_scam']} scam)", f"{c['english_scripts_rows']:,}")

    t1, t2, t3, t4, t5 = st.tabs(["Sources & cleaning", "Length", "Vocabulary & phrases", "Correlations", "Samples"])
    with t1:
        d = m.get("data", {})
        if d:
            rows = []
            for k in ("composite_train", "composite_test", "english_scripts", "fraud_call"):
                r = d[k]
                rows.append({"Source": k, "Raw rows": r["raw_rows"], "Empty after cleaning": r["empty_after_cleaning"],
                             "Conflicting labels removed": r["conflicting_label_texts_removed"], "Duplicates removed": r["duplicates_removed"],
                             "Final rows": r["final_rows"], "Scam": r["scam"], "Overlap with composite_train": r.get("overlap_with_composite_train")})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            sp = d["splits"]
            st.markdown(f"**How each source is used.** Training = composite 80% ({sp['main_train']['rows']:,}) + half the English scripts "
                        f"({sp['ext_train']['rows']}). Tests = composite 20% ({sp['main_test']['rows']:,}), other half of English scripts "
                        f"({sp['ext_test']['rows']}), composite_test rows not in training ({sp['composite_test_unseen']['rows']:,}) and "
                        f"fraud_call rows not in training ({sp['fraud_call_unseen']['rows']}).")
        fig = go.Figure(go.Bar(x=["Non-scam", "Scam"], y=[c["non_scam"], c["scam"]], marker=dict(color=[C["green"], C["red"]]),
                               text=[f"{c['non_scam']:,}", f"{c['scam']:,}"], textposition="outside", width=0.5,
                               hovertemplate="%{x}: %{y:,}<extra></extra>"))
        fig.update_layout(title="Class distribution (composite, after cleaning)", yaxis_title="Conversations", showlegend=False)
        plot(fig, height=320, key="classdist")
        note(f"Moderately imbalanced: {pct(c['scam_share'])} scam. A model that always answers 'scam' would already be {pct(c['scam_share'])} accurate, "
             "so accuracy alone is a weak metric here — F1 and recall are reported throughout. Stratified splitting keeps this ratio in every split.")
        art = e["artifact_tokens"]
        st.markdown("**Source artifacts and missing India-specific vocabulary** (rows containing the word, raw composite_train.csv)")
        st.dataframe(pd.DataFrame([{"Word": w, "Scam rows": v["scam"], "Non-scam rows": v["non_scam"]} for w, v in art.items()]),
                     hide_index=True, width="stretch")
        note("Translated speaker names (zhang, xiao, li) appear only in scam rows and are removed during preprocessing. Terms typical of Indian "
             "scams (OTP, KYC, UPI, Aadhaar, AnyDesk) are almost absent, so India-specific red flags are handled by the transparent rule layer.", "warn")
    with t2:
        L = e["length"]
        st.dataframe(pd.DataFrame([{"Class": "Scam" if k == "scam" else "Non-scam", "Mean words": f"{v['word_count']['mean']:.0f}",
                                    "Median words": f"{v['word_count']['median']:.0f}", "25th–75th pct": f"{v['word_count']['p25']:.0f}–{v['word_count']['p75']:.0f}",
                                    "95th pct": f"{v['word_count']['p95']:.0f}", "Max": f"{v['word_count']['max']:.0f}"} for k, v in L.items()]),
                     hide_index=True, width="stretch")
        log_scale = st.toggle("Logarithmic word-count axis", value=True)
        h = e["word_hist_log"] if log_scale else e["word_hist"]
        edges = h["edges"]
        mids = [(edges[i] * edges[i + 1]) ** 0.5 if log_scale else (edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mids, y=h["non_scam"], name="Non-scam", mode="lines", line=dict(color=C["s1"], width=2, shape="hvh"),
                                 hovertemplate="~%{x:.0f} words: %{y:,}<extra>Non-scam</extra>"))
        fig.add_trace(go.Scatter(x=mids, y=h["scam"], name="Scam", mode="lines", line=dict(color=C["s2"], width=2, shape="hvh"),
                                 hovertemplate="~%{x:.0f} words: %{y:,}<extra>Scam</extra>"))
        fig.update_layout(title="Word-count distribution by class" + ("" if log_scale else f" (clipped at 99th percentile, {h['cap_p99']:.0f} words)"),
                          xaxis=dict(type="log" if log_scale else "linear", title="Words per conversation"), yaxis_title="Conversations",
                          hovermode="x unified")
        plot(fig, height=360, key="hist")
        note(f"<b>Corrected interpretation.</b> The original EDA said scam conversations are 'more concise' because their <i>mean</i> is lower "
             f"({L['scam']['word_count']['mean']:.0f} vs {L['non_scam']['word_count']['mean']:.0f} words). The mean is pulled up by a long tail of very long "
             f"non-scam transcripts. The <i>median</i> scam conversation is far longer ({L['scam']['word_count']['median']:.0f} vs "
             f"{L['non_scam']['word_count']['median']:.0f} words) because many non-scam rows are short SMS messages. Use medians for skewed data.")
    with t3:
        a, b = st.columns(2)
        with a:
            _bar_terms(e["top_bigrams"]["scam"], "Most frequent bigrams — scam (documents)", C["s2"], "bg_s")
        with b:
            _bar_terms(e["top_bigrams"]["non_scam"], "Most frequent bigrams — non-scam (documents)", C["s1"], "bg_n")
        a, b = st.columns(2)
        with a:
            _bar_terms(e["top_unigrams"]["scam"], "Most frequent words — scam", C["s2"], "ug_s")
        with b:
            _bar_terms(e["top_unigrams"]["non_scam"], "Most frequent words — non-scam", C["s1"], "ug_n")
        st.markdown("**Most class-distinctive terms** (smoothed log-odds of document frequency, min. 30 documents)")
        a, b = st.columns(2)
        a.dataframe(pd.DataFrame([{"Scam-distinctive": t, "log-odds": round(s, 2), "scam docs": x, "non-scam docs": y} for t, s, x, y in e["distinctive"]["scam"]]),
                    hide_index=True, width="stretch", height=380)
        b.dataframe(pd.DataFrame([{"Non-scam-distinctive": t, "log-odds": round(s, 2), "non-scam docs": x, "scam docs": y} for t, s, x, y in e["distinctive"]["non_scam"]]),
                    hide_index=True, width="stretch", height=380)
        v = e["vocabulary"]
        st.caption(f"Vocabulary after cleaning — scam: {v['scam']['vocabulary']:,} distinct words over {v['scam']['tokens']:,} tokens; "
                   f"non-scam: {v['non_scam']['vocabulary']:,} distinct words over {v['non_scam']['tokens']:,} tokens. Scam dialogues are "
                   "far more repetitive (templated scripts), non-scam text is more varied. Stop-words are excluded from these charts but kept for modelling.")
        note("Several non-scam-distinctive terms (e.g. topical words from casual-chat transcripts) describe the non-scam <i>sources</i>, not the absence "
             "of fraud. This is another face of the dataset shortcut.", "warn")
    with t4:
        kind = st.radio("Correlation", ["spearman", "pearson"], horizontal=True,
                        format_func=lambda s: "Spearman (rank, robust to outliers)" if s == "spearman" else "Pearson (linear)")
        cols = e["corr"]["columns"]
        z = e["corr"][kind]
        names = [s.replace("_", " ") for s in cols]
        fig = go.Figure(go.Heatmap(z=z, x=names, y=names, colorscale=DIV_SCALE, zmin=-1, zmax=1, text=[[f"{v:.2f}" for v in r] for r in z],
                                   texttemplate="%{text}", textfont=dict(color=C["text"]), hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>",
                                   colorbar=dict(thickness=10)))
        fig.update_layout(title=f"{kind.title()} correlation matrix", yaxis=dict(autorange="reversed"))
        plot(fig, height=480, key="corr")
        p0, s0 = e["corr"]["pearson"][0], e["corr"]["spearman"][0]
        note(f"<b>Corrected interpretation.</b> The original EDA concluded length is weakly related to the label (Pearson r = {p0[1]:.2f}). "
             f"Pearson only measures <i>linear</i> association and is distorted by the long tail. The rank correlation is {s0[1]:+.2f}, and a "
             "depth-3 decision tree on word count alone is ≈86% accurate in the audit. Average word length is even more associated with the label "
             f"(Spearman {s0[4]:+.2f}). These are properties of how the dataset was assembled, not of scams in general.")
    with t5:
        st.caption("Random examples (25–400 words). Phone numbers, e-mail addresses and links are masked; text truncated to 500 characters.")
        for s in e["samples"]:
            lab = "Scam" if s["label"] else "Non-scam"
            col = C["red"] if s["label"] else C["green"]
            src = " · English scripts" if s.get("source") else ""
            st.markdown(f'<div class="panel thin"><div style="color:{col};font-weight:600;font-size:.9rem">{lab}{src} — {s["words"]} words</div>'
                        f'<div style="color:var(--text2);font-size:.92rem;margin-top:4px">{esc(s["text"])}</div></div>', unsafe_allow_html=True)
