"""Page 3 — Model comparison (all numbers read from reports/metrics.json, nothing hard-coded)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ui import C, SEQ_SCALE, load_json, note, pct, plot


def render():
    st.title("Model comparison")
    m = load_json("metrics.json")
    if not m:
        st.error("reports/metrics.json not found — run `python scripts/train_models.py`.")
        return
    models = m["models"]
    best = m["best_model"]
    st.markdown(f'<p class="lede">Eight classifiers share one TF-IDF representation ({m["vocabulary_size"]:,} uni- and bigrams). '
                f'The default model is chosen by <b>{m["selection_metric"][0].lower() + m["selection_metric"][1:]}</b>, so no test set influenced the choice. '
                f'Selected: <b>{models[best]["info"]["name"]}</b>.</p>', unsafe_allow_html=True)

    sets = m["eval_sets"]
    set_key = st.radio("Evaluate on", list(sets), format_func=sets.get, horizontal=True)
    rows = []
    for k, r in models.items():
        t = r["test"][set_key]
        rows.append({"key": k, "Model": r["info"]["name"] + (" ★" if k == best else ""),
                     "CV F1 (train, 5-fold)": r["cv"]["f1"]["mean"] if r.get("cv") else None,
                     "Accuracy": t["accuracy"], "Precision": t["precision"], "Recall": t["recall"], "F1": t["f1"],
                     "ROC-AUC": t.get("roc_auc"), "Brier": t.get("brier"), "False negatives": t["false_negatives"],
                     "False positives": t["false_positives"], "Fit time (s)": r["fit_seconds"]})
    df = pd.DataFrame(rows).sort_values("F1", ascending=False)
    for col in ("CV F1 (train, 5-fold)", "ROC-AUC", "Brier"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    n = models[best]["test"][set_key]["n"]
    st.caption(f"{sets[set_key]} — n = {n:,} conversations. ★ = default model. Brier is blank for hard voting (no probabilities).")
    show = df.drop(columns="key").copy()
    for c in ("CV F1 (train, 5-fold)", "Accuracy", "Precision", "Recall", "F1"):
        show[c] = show[c].map(lambda v: "—" if pd.isna(v) else f"{v:.2%}")
    for c, f in (("ROC-AUC", "{:.4f}"), ("Brier", "{:.4f}"), ("Fit time (s)", "{:.1f}")):
        show[c] = show[c].map(lambda v, f=f: "—" if pd.isna(v) else f.format(v))
    st.dataframe(show, hide_index=True, width="stretch")

    # ---- F1 and recall dot plot (two series, validated categorical slots 1-2)
    d2 = df.sort_values("F1")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d2["F1"], y=d2["Model"], mode="markers", name="F1", marker=dict(size=12, color=C["s1"], line=dict(width=2, color=C["bg"])),
                             hovertemplate="%{y}<br>F1 %{x:.2%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=d2["Recall"], y=d2["Model"], mode="markers", name="Recall (scam caught)", marker=dict(size=10, symbol="diamond", color=C["s2"], line=dict(width=2, color=C["bg"])),
                             hovertemplate="%{y}<br>Recall %{x:.2%}<extra></extra>"))
    lo = max(0.0, min(d2["F1"].min(), d2["Recall"].min()) - 0.02)
    fig.update_layout(title="F1 and recall by model", xaxis=dict(tickformat=".0%", range=[lo, 1.005]), yaxis=dict(automargin=True))
    plot(fig, height=380, key="f1dot")
    note("The x-axis is zoomed to the data range to make small differences visible. On the in-distribution test, the top linear and "
         "ensemble models are within a fraction of a percentage point of each other — differences that small are within cross-validation noise "
         "(see the ± in the CV column below).")

    # ---- confusion matrices
    st.subheader("Confusion matrices")
    keys = list(models)
    fig = make_subplots(rows=2, cols=4, subplot_titles=[models[k]["info"]["name"].split(" (")[0] for k in keys],
                        horizontal_spacing=0.06, vertical_spacing=0.18)
    for i, k in enumerate(keys):
        cm = models[k]["test"][set_key]["confusion_matrix"]
        tot = sum(map(sum, cm))
        fig.add_trace(go.Heatmap(z=cm, x=["Pred. non-scam", "Pred. scam"], y=["Actual non-scam", "Actual scam"], colorscale=SEQ_SCALE,
                                 showscale=False, text=[[f"{v:,}<br>{v/tot:.1%}" for v in row] for row in cm], texttemplate="%{text}",
                                 textfont=dict(color=C["text"], size=11), hovertemplate="%{y} / %{x}: %{z:,}<extra></extra>"),
                      row=i // 4 + 1, col=i % 4 + 1)
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=10))
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_annotations(font=dict(size=12, color=C["text"]))
    plot(fig, height=470, key="cms")
    note("<b>Why false negatives matter most:</b> a false negative is a scam the system calls safe — the user relaxes exactly when they "
         "should not. A false positive costs a moment of extra caution. That asymmetry is why the MEDIUM risk band starts below 50% "
         "and why recall is plotted next to F1.", "warn")

    # ---- ROC / PR / calibration
    st.subheader("Threshold-free and calibration views")
    sel = st.multiselect("Models to plot (max 3)", keys, default=[best, "naive_bayes", "logistic_regression"][:3],
                         format_func=lambda k: models[k]["info"]["name"], max_selections=3)
    slot = [C["s1"], C["s2"], C["s3"]]
    a, b, c = st.columns(3)
    figs = [go.Figure(), go.Figure(), go.Figure()]
    for j, k in enumerate(sel):
        t = models[k]["test"][set_key]
        nm = models[k]["info"]["name"].split(" (")[0]
        if "roc_curve" in t:
            figs[0].add_trace(go.Scatter(x=t["roc_curve"]["fpr"], y=t["roc_curve"]["tpr"], name=f"{nm} (AUC {t['roc_auc']:.3f})",
                                         line=dict(width=2, color=slot[j])))
            figs[1].add_trace(go.Scatter(x=t["pr_curve"]["recall"], y=t["pr_curve"]["precision"], name=f"{nm} (AP {t['average_precision']:.3f})",
                                         line=dict(width=2, color=slot[j])))
        if "reliability" in t:
            rel = t["reliability"]
            figs[2].add_trace(go.Scatter(x=[d["mean_pred"] for d in rel], y=[d["frac_pos"] for d in rel], mode="lines+markers",
                                         name=f"{nm} (Brier {t['brier']:.3f}, ECE {t['ece']:.3f})", line=dict(width=2, color=slot[j]),
                                         marker=dict(size=8), customdata=[d["n"] for d in rel],
                                         hovertemplate="predicted %{x:.2f}<br>observed %{y:.2f}<br>n=%{customdata}<extra></extra>"))
    figs[0].add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dot", color=C["muted"], width=1), showlegend=False, hoverinfo="skip"))
    figs[2].add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dot", color=C["muted"], width=1), name="perfect calibration", hoverinfo="skip"))
    for f, ttl, xt, yt in [(figs[0], "ROC curve", "False positive rate", "True positive rate"),
                           (figs[1], "Precision–recall curve", "Recall", "Precision"),
                           (figs[2], "Reliability (calibration)", "Predicted P(scam)", "Observed scam share")]:
        f.update_layout(title=ttl, xaxis_title=xt, yaxis_title=yt, legend=dict(orientation="h", y=-0.3, font=dict(size=11)))
    for col, f, kk in zip((a, b, c), figs, ("roc", "pr", "rel")):
        with col:
            plot(f, height=400, key=kk)
    note("A well-calibrated model's dots lie on the diagonal: when it says 80%, about 80% of such conversations are scams. "
         "Naive Bayes is typically over-confident (dots far from the diagonal), which is why its percentages are not shown as the default risk.")

    # ---- CV table
    st.subheader("Cross-validation on the training split (model selection)")
    cv = pd.DataFrame([{"Model": r["info"]["name"], **{f"{mm} (mean ± sd)": f"{r['cv'][mm]['mean']*100:.2f} ± {r['cv'][mm]['std']*100:.2f}"
                                                      for mm in ("accuracy", "precision", "recall", "f1")}, "CV time (s)": r["cv"]["seconds"]}
                       for k, r in models.items() if r.get("cv")])
    st.dataframe(cv, hide_index=True, width="stretch")

    # ---- ablations
    st.subheader("Why the scores look the way they do")
    ab = m["ablations"]
    h = ab["historical_reproduction"]
    st.markdown("**1. Historical notebook results — reproduced** (original cleaning, composite data only, default hyper-parameters)")
    st.dataframe(pd.DataFrame([{"Model": k, "Reported in notebook": pct(v["reported_accuracy"], 2), "Reproduced accuracy": pct(v["reproduced_accuracy"], 2),
                                "Reproduced F1": pct(v["reproduced_f1"], 2)} for k, v in h["models"].items()]), hide_index=True, width="stretch")
    st.caption(f"{h['rows']:,} rows after the original cleaning ({h['scam']:,} scam). Differences ≤ 0.2 points come from a regex detail in the missing cleaning script.")

    f = ab["format_length_shortcut"]
    st.markdown("**2. Dataset shortcut** — a Random Forest that sees only raw length, word count, quote/comma format and question marks (no words):")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy without reading any words", pct(f["format_length_only_rf_accuracy"]))
    c2.metric("Majority-class baseline", pct(f["majority_baseline_accuracy"]))
    c3.metric(f"List-format rows: scam vs non-scam", f"{pct(f['list_format_share']['scam'], 0)} vs {pct(f['list_format_share']['non_scam'], 0)}")
    note("Most of the composite dataset's scam rows come from a different source and format than its non-scam rows. High in-distribution "
         "accuracy therefore partly measures <i>which source a row came from</i>. This is the main reason we also report an external test.", "warn")

    o = ab["out_of_domain_english"]
    st.markdown("**3. Out-of-domain test** — trained on composite data only, tested on all English scripts (never seen):")
    st.dataframe(pd.DataFrame([{"Model": models[k]["info"]["name"], "Accuracy": pct(v["accuracy"]), "Recall": pct(v["recall"]),
                                "Precision": pct(v["precision"]), "F1": pct(v["f1"])} for k, v in o.items()]), hide_index=True, width="stretch")
    ext_best = models[best]["test"]["ext_test"]
    st.caption(f"After adding half of the English scripts to training, the default model reaches {pct(ext_best['accuracy'])} accuracy "
               f"and {pct(ext_best['recall'])} recall on the other (held-out) half. Both halves come from the same template style, so this "
               "is still an optimistic estimate for real calls.")

    mk = ab["artifact_masking"]
    st.markdown("**4. Masking source-identifying names** (Logistic Regression):")
    st.dataframe(pd.DataFrame([{"Preprocessing": "Names masked (final)" if k == "masked" else "Names kept",
                                "In-distribution accuracy": pct(v["main_test"]["accuracy"], 2), "External accuracy": pct(v["ext_test"]["accuracy"], 2),
                                "External recall": pct(v["ext_test"]["recall"], 2)} for k, v in mk.items()]), hide_index=True, width="stretch")
    nd = ab["near_duplicates"]
    st.markdown(f"**5. Near-duplicates:** {pct(nd['share_of_main_test_with_train_neighbour'])} of in-distribution test conversations have a "
                f"training neighbour with cosine similarity ≥ 0.90. Accuracy of the default model on the test set *without* them: "
                f"{pct(models[best]['test']['main_test_no_near_duplicates']['accuracy'], 2)} — so near-duplicates do not explain the high score.")

    ts = load_json("tuning_study.json")
    if ts:
        with st.expander("Hyper-parameter study (3-fold CV, training split only)"):
            st.dataframe(pd.DataFrame(ts["rows"]).style.format({"cv_f1": "{:.4f}", "cv_recall": "{:.4f}", "cv_precision": "{:.4f}"}),
                         hide_index=True, width="stretch")
            st.caption("Result: the original TF-IDF setting (10k features, 1–2 grams) was kept; LR C=10 and NB α=0.1 were adopted.")

    with st.expander("Regression experiments from the original notebook (why they are not used for risk)"):
        st.markdown("Linear, Ridge and Lasso regression were fitted to the 0/1 label in the original notebook "
                    "(reported RMSE 0.164 / 0.130 / 0.281). They are kept as an academic comparison only:\n"
                    "- predictions fall outside [0, 1] (the notebook shows values such as 1.22 and −0.16), so they are not probabilities;\n"
                    "- squared error on a binary target rewards being close to 0.5 rather than ranking scams correctly;\n"
                    "- Logistic Regression is the principled version of the same idea — a linear score passed through a sigmoid and fitted by log-loss.")
