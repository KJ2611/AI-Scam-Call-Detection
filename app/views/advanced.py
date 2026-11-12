"""Page 5 — Advanced ML analysis: clustering and Hidden Markov Models (supplementary)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from scam_analyzer.preprocessing import pretty_token
from ui import C, SEQ_SCALE, load_json, note, pct, plot


def _three_paradigms():
    st.markdown("""
| | Supervised classification | Unsupervised clustering | Sequential modelling (HMM) |
|---|---|---|---|
| **Uses labels?** | Yes — learns P(scam \\| words) | No — groups similar texts | Labels only to pick which sequences train which HMM |
| **Question answered** | "Is this a scam?" | "Which conversations look alike?" | "In what order do things happen?" |
| **Output** | Class + probability | Cluster ID (arbitrary number) | Hidden-state path, sequence likelihood |
| **Role in this project** | **The detector** used by the app | Exploration of structure / topics | Supplementary stage analysis |
""")
    note("A cluster ID is not a label: “cluster 1” does not mean “scam”. Agreement with the real labels is measured with ARI / NMI / purity, "
         "and it is low — which is expected, because clustering groups by <i>topic and style</i>, not by fraud intent.")


def _clustering(cl):
    rows = pd.DataFrame(cl["results"])
    show = rows[["method", "sizes", "silhouette", "davies_bouldin", "calinski_harabasz", "ari", "nmi", "purity", "degenerate"]].copy()
    show["sizes"] = show["sizes"].map(lambda s: " / ".join(f"{x:,}" for x in s))
    show.columns = ["Method", "Cluster sizes", "Silhouette ↑", "Davies–Bouldin ↓", "Calinski–Harabasz ↑", "ARI vs label ↑", "NMI vs label ↑", "Purity", "Degenerate"]
    st.dataframe(show.style.format({"Silhouette ↑": "{:.3f}", "Davies–Bouldin ↓": "{:.3f}", "Calinski–Harabasz ↑": "{:.0f}",
                                    "ARI vs label ↑": "{:.3f}", "NMI vs label ↑": "{:.3f}", "Purity": "{:.1%}"}),
                 hide_index=True, width="stretch")
    st.caption(f"k = 2 for every method, {cl['sample_size']:,}-conversation stratified sample. Feature space for all methods and all metrics: {cl['feature_space']}.")
    best_ari = rows.loc[rows.ari.idxmax()]
    best_sil = rows.loc[rows[~rows.degenerate].silhouette.idxmax()]
    note(f"Best label agreement: <b>{best_ari['method']}</b> (ARI {best_ari['ari']:.3f}, purity {best_ari['purity']:.1%}). "
         f"Best-separated clusters: <b>{best_sil['method']}</b> (silhouette {best_sil['silhouette']:.3f}). Purity must be compared with the "
         f"{pct(cl['majority_share'])} you get by putting everything in one cluster.")

    p = cl["plot"]
    method = st.selectbox("Colour the 2-D projection by", ["True label"] + list(p["clusters"]) + ["K-Means, k = 8 (topics)"])
    if method == "True label":
        lab, names = p["label"], {0: "Non-scam", 1: "Scam"}
    elif method.startswith("K-Means, k = 8"):
        lab, names = p["kmeans8"], {i: f"Topic cluster {i}" for i in range(8)}
    else:
        lab, names = p["clusters"][method], {0: "Cluster 0", 1: "Cluster 1"}
    fig = go.Figure()
    palette = [C["s1"], C["s2"], C["s3"], "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
    for i, g in enumerate(sorted(set(lab))):
        idx = [j for j, v in enumerate(lab) if v == g]
        fig.add_trace(go.Scattergl(x=[p["x"][j] for j in idx], y=[p["y"][j] for j in idx], mode="markers", name=names.get(g, str(g)),
                                   marker=dict(size=6, color=palette[i % 8], opacity=.75, line=dict(width=0.5, color=C["bg"]))))
    fig.update_layout(title=f"TF-IDF projected to 2 SVD components — coloured by {method.lower()}", xaxis_title="SVD component 1",
                      yaxis_title="SVD component 2")
    plot(fig, height=460, key="scatter")
    st.caption("The 2-D projection is for viewing only; clustering itself ran in 50 dimensions.")

    a, b = st.columns(2)
    with a:
        kc = pd.DataFrame(cl["k_curve"])
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        fig.add_trace(go.Scatter(x=kc.k, y=kc.silhouette, mode="lines+markers", line=dict(color=C["s1"], width=2), marker=dict(size=8),
                                 name="silhouette", hovertemplate="k=%{x}: %{y:.3f}<extra></extra>"))
        fig.update_layout(title="Choosing k for K-Means: silhouette score", xaxis_title="k", yaxis_title="Silhouette", showlegend=False)
        plot(fig, height=320, key="kcurve")
    with b:
        fig = go.Figure(go.Scatter(x=kc.k, y=kc.inertia, mode="lines+markers", line=dict(color=C["s2"], width=2), marker=dict(size=8),
                                   hovertemplate="k=%{x}: %{y:.0f}<extra></extra>"))
        fig.update_layout(title="Elbow method: inertia (within-cluster SS)", xaxis_title="k", yaxis_title="Inertia")
        plot(fig, height=320, key="elbow")

    st.markdown("**What the clusters are about** — K-Means with k = 8, sorted by share of scam conversations")
    st.dataframe(pd.DataFrame([{"Cluster": t["cluster"], "Size": t["size"], "Scam share": t["scam_share"], "Top terms (centroid)": ", ".join(pretty_token(x) for x in t["top_terms"])}
                               for t in cl["kmeans_topics"]]).style.format({"Scam share": "{:.0%}"}), hide_index=True, width="stretch")
    note("This is the most useful output of clustering for the project: unsupervised topics such as bank-verification, prize/lottery, or casual chat "
         "emerge without labels, and the scam share per topic shows which themes the scam data is built from.")
    with st.expander("Issues found in the original clustering notebook and how they were fixed"):
        for i in cl["original_notebook"]["issues"]:
            st.markdown(f"- {i}")


def _hmm(h):
    syms = h["symbols"]
    st.markdown(f"""
**Set-up.** Each cleaned transcript is cut into windows of {h['window_tokens']} words. Every window becomes one *observation symbol* — the most specific
manipulation category whose keywords it contains (e.g. AUTHORITY, ACCOUNT, URGENCY, CREDENTIAL, PAYMENT) or NEUTRAL. Two {h['n_states']}-state
categorical HMMs are trained with **Baum–Welch** (EM): one on {h['train_sequences']['scam']:,} scam sequences, one on {h['train_sequences']['non_scam']:,}
non-scam sequences. A new transcript is assigned to the class whose HMM gives the higher likelihood (computed with the **Forward** algorithm).
""")
    a, b = st.columns([1, 1.2])
    with a:
        f = h["symbol_frequency"]
        fig = go.Figure()
        fig.add_trace(go.Bar(y=syms, x=f["non_scam"], name="Non-scam", orientation="h", marker=dict(color=C["s1"])))
        fig.add_trace(go.Bar(y=syms, x=f["scam"], name="Scam", orientation="h", marker=dict(color=C["s2"])))
        fig.update_layout(title="Observation symbols: share of windows", xaxis=dict(tickformat=".0%"), barmode="group", bargap=.3,
                          yaxis=dict(autorange="reversed", automargin=True))
        plot(fig, height=420, key="symfreq")
    with b:
        ev = h["evaluation"]
        rows = []
        for split, label in (("main_test", "In-distribution test"), ("ext_test", "External test")):
            for k, nm in (("hmm", "Class-conditional HMMs (order matters)"), ("bag_of_symbols_nb", "Naive Bayes on symbol counts (order ignored)")):
                rows.append({"Test set": label, "Model": nm, "Accuracy": ev[split][k]["accuracy"], "F1": ev[split][k]["f1"], "Recall": ev[split][k]["recall"]})
            rows.append({"Test set": label, "Model": "Majority class", "Accuracy": ev[split]["majority_baseline"], "F1": None, "Recall": None})
        st.dataframe(pd.DataFrame(rows).style.format({"Accuracy": "{:.1%}", "F1": "{:.1%}", "Recall": "{:.1%}"}, na_rep="—"),
                     hide_index=True, width="stretch")
        d_in = ev["main_test"]["hmm"]["accuracy"] - ev["main_test"]["bag_of_symbols_nb"]["accuracy"]
        note(f"Comparing the HMM with the order-blind baseline isolates the value of <i>sequence</i>: a difference of {d_in*100:+.1f} points in-distribution. "
             "Both are far below the TF-IDF classifiers — eleven coarse symbols carry much less information than 10,000 n-grams — so the HMM is used "
             "for explanation of conversation flow, not for the verdict.")

    names = h.get("state_names") or [f"State {i}" for i in range(h["n_states"])]
    a, b = st.columns(2)
    for col, key, ttl in ((a, "scam_hmm", "Scam HMM"), (b, "non_scam_hmm", "Non-scam HMM")):
        with col:
            P = h[key]
            nm = names if key == "scam_hmm" else (h.get("state_names_non") or [f"State {i}" for i in range(h["n_states"])])
            fig = go.Figure(go.Heatmap(z=P["transmat"], x=nm, y=nm, colorscale=SEQ_SCALE, zmin=0, zmax=1,
                                       text=[[f"{v:.2f}" for v in r] for r in P["transmat"]], texttemplate="%{text}",
                                       textfont=dict(color=C["text"]), hovertemplate="from %{y} to %{x}: %{z:.3f}<extra></extra>", showscale=False))
            fig.update_layout(title=f"{ttl} — transition matrix A (row = from)", yaxis=dict(autorange="reversed", automargin=True), xaxis=dict(automargin=True))
            plot(fig, height=330, key=f"tm_{key}")
            fig = go.Figure(go.Heatmap(z=P["emissionprob"], x=syms, y=nm, colorscale=SEQ_SCALE, zmin=0,
                                       hovertemplate="%{y} emits %{x}: %{z:.3f}<extra></extra>", colorbar=dict(thickness=8)))
            fig.update_layout(title=f"{ttl} — emission matrix B", yaxis=dict(autorange="reversed", automargin=True), xaxis=dict(tickangle=-40))
            plot(fig, height=320, key=f"em_{key}")
            st.caption("Initial probabilities π: " + ", ".join(f"{n} {p:.2f}" for n, p in zip(nm, P["startprob"])))

    v = h["verification"]
    st.markdown("**Algorithms implemented from scratch** (`src/scam_analyzer/hmm_algorithms.py`) and checked against `hmmlearn`:")
    c1, c2, c3 = st.columns(3)
    c1.metric("Max |log-likelihood difference|", f"{v['max_abs_loglik_diff']:.2e}", help="Scaled Forward algorithm vs hmmlearn.score()")
    c2.metric("Viterbi paths identical", f"{v['viterbi_paths_identical']} / {v['n_checked']}")
    u = v["underflow_demo"]
    c3.metric(f"T = {u['length']:,}: unscaled P vs scaled log P", f"{u['unscaled_probability']:.0e} vs {u['scaled_log_likelihood']:.0f}")
    st.markdown(r"""
- **Forward:** $\alpha_t(j) = \big[\sum_i \alpha_{t-1}(i)\,a_{ij}\big]\, b_j(o_t)$, normalised each step by $c_t$; $\log P(O\mid\lambda)=\sum_t \log c_t$.
- **Backward:** $\beta_t(i) = \sum_j a_{ij}\, b_j(o_{t+1})\, \beta_{t+1}(j)$, scaled with the same $c_{t+1}$.
- **Forward–Backward posterior:** $\gamma_t(i) = \alpha_t(i)\beta_t(i) / \sum_j \alpha_t(j)\beta_t(j)$ — used for the live state plot on the detection page.
- **Viterbi:** $\delta_t(j) = \max_i[\delta_{t-1}(i) + \log a_{ij}] + \log b_j(o_t)$ with back-pointers — the single most likely state path.
- **Baum–Welch:** EM that re-estimates π, A, B from expected counts computed with Forward–Backward (via `hmmlearn`, 3 random restarts).
""")
    with st.expander("Issues found in the original HMM notebook and how they were fixed"):
        for i in h["original_notebook"]["issues"]:
            st.markdown(f"- {i}")
        st.markdown("- **Gaussian HMM:** not re-run. A Gaussian HMM needs a genuine time series of continuous features *within* each conversation; "
                    "fitting it across unrelated rows is not meaningful. The discrete class-conditional HMM above replaces it.")


def render():
    st.title("Advanced ML analysis")
    st.markdown('<p class="lede">Supplementary experiments that keep the clustering and HMM work from the original notebooks, re-run with the '
                'methodological fixes found in the audit. None of these feed the verdict on the detection page.</p>', unsafe_allow_html=True)
    adv = load_json("advanced.json")
    if not adv:
        st.error("reports/advanced.json not found — run `python scripts/run_analysis.py`.")
        return
    t0, t1, t2 = st.tabs(["Supervised vs unsupervised vs sequential", "Clustering", "Hidden Markov Models"])
    with t0:
        _three_paradigms()
    with t1:
        _clustering(adv["clustering"])
    with t2:
        _hmm(adv["hmm"])
