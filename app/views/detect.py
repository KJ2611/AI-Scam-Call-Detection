"""Page 2 — Scam Detection (the live demo)."""
from __future__ import annotations

import re

import plotly.graph_objects as go
import streamlit as st

from examples import EXAMPLES
from scam_analyzer.models import MODEL_INFO
from ui import (C, RISK_COLOR, RISK_ICON, esc, get_analyzer, get_hmm_bundle, load_json, missing_models_message,
                models_ready, note, plot)


def _set_text(t: str):
    st.session_state["transcript"] = t
    st.session_state.pop("result", None)


def _highlight(raw: str, result: dict) -> str:
    """Wrap rule-indicator spans (red) and model-evidence terms (cyan / green) in <mark>."""
    spans = []  # (start, end, cls)
    for d in result["indicators"]["indicators"]:
        for s, e in d["spans"]:
            spans.append((s, e, "r"))

    def add_terms(terms, cls):
        for term, *_ in terms:
            if term.startswith("<"):
                continue
            words = term.split()
            rx = re.compile(r"\b" + r"[\W_]+".join(map(re.escape, words)) + r"\b", re.I)
            for m in rx.finditer(raw):
                spans.append((m.start(), m.end(), cls))
    add_terms(result["evidence"]["toward_scam"][:10], "m")
    add_terms(result["evidence"]["toward_non_scam"][:8], "g")
    # resolve overlaps: rule spans win, then longer spans
    spans.sort(key=lambda x: (x[0], 0 if x[2] == "r" else 1, -(x[1] - x[0])))
    out, pos = [], 0
    for s, e, cls in spans:
        if s < pos:
            continue
        out.append(esc(raw[pos:s]))
        out.append(f'<mark class="{cls}">{esc(raw[s:e])}</mark>')
        pos = e
    out.append(esc(raw[pos:]))
    return "".join(out)


def _fmt_prob(p: float) -> str:
    """Avoid displaying a calibrated estimate as certainty."""
    if p >= 0.999:
        return "&gt;99.9%"
    if p <= 0.001:
        return "&lt;0.1%"
    return f"{p*100:.1f}%"


def _verdict(r: dict):
    risk, p = r["risk"], r["score"]
    col = RISK_COLOR[risk]
    th = r["thresholds"]
    pred_col = C["red"] if r["prediction"] == "SCAM" else C["green"]
    prob_label = "estimated probability of scam" if r["probabilistic"] else "share of voting models that said scam"
    unc = " The probability lies between the two thresholds, so the model is <b>uncertain</b> here." if r["uncertain"] else ""
    st.markdown(f"""
<div class="verdict" style="border-color:{col}55; box-shadow: inset 4px 0 0 {col};">
  <div style="display:flex; justify-content:space-between; align-items:flex-end; gap:24px; flex-wrap:wrap;">
    <div>
      <div class="label" style="color:{pred_col}">{r['prediction']}</div>
      <div class="sub">Risk level <b style="color:{col}">{RISK_ICON[risk]} {risk}</b>, scored by {esc(r['model_name'])}</div>
    </div>
    <div style="text-align:right">
      <div class="big-prob" style="color:{col}">{_fmt_prob(p)}</div>
      <div class="sub">{prob_label}</div>
    </div>
  </div>
  <div class="meter" style="--m:{th['medium']*100}%; --h:{th['high']*100}%;" role="img"
       aria-label="Probability {p*100:.1f} percent; low below {th['medium']*100:.0f}, high from {th['high']*100:.0f}">
    <div class="needle" style="left:calc({min(max(p,0),1)*100}% - 2px)"></div>
  </div>
  <div class="meter-scale"><span style="left:0">0%</span><span style="left:{th['medium']*100}%">Medium from {th['medium']*100:.0f}%</span><span style="left:{th['high']*100}%">High from {th['high']*100:.0f}%</span><span style="right:0">100%</span></div>
  <div class="sub" style="margin-top:10px">{esc(r['disclaimer'])}{unc}</div>
</div>""", unsafe_allow_html=True)
    if r["escalation"]:
        note(esc(r["escalation"]), "warn")
    for w in r["warnings"]:
        note(esc(w), "warn")


def _evidence_chart(ev: dict):
    pos = ev["toward_scam"][:10][::-1]
    neg = ev["toward_non_scam"][:10][::-1]
    terms = [t for t, *_ in neg] + [t for t, *_ in pos]
    vals = [c for _, c, *_ in neg] + [c for _, c, *_ in pos]
    if not terms:
        st.caption("No known terms — the model had no word-level evidence.")
        return
    colors = [C["red"] if v > 0 else C["green"] for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=terms, orientation="h", marker=dict(color=colors, line=dict(width=0)),
                           hovertemplate="%{y}<br>contribution %{x:+.3f} " + ev["units"] + "<extra></extra>"))
    fig.update_layout(title=f"Model evidence — contribution to the {ev['units']}",
                      xaxis_title="← toward legitimate      toward scam →", yaxis=dict(automargin=True),
                      bargap=0.35, showlegend=False)
    plot(fig, height=max(260, 26 * len(terms) + 90), key="evidence")
    st.caption(f"Source: {esc(ev['source'])}. Contribution = TF-IDF value × learned weight; these add up exactly to the model score "
               f"(total toward scam {ev['sum_scam']:+.2f}, toward legitimate {ev['sum_non_scam']:+.2f}, {ev['n_terms']} known terms).")


def _hmm_trace(cleaned: str):
    bundle = get_hmm_bundle()
    if bundle is None:
        return
    from scam_analyzer.advanced import decode_transcript
    d = decode_transcript(cleaned, bundle)
    if d is None or len(d["symbols"]) < 2:
        st.caption("Transcript too short for a stage sequence.")
        return
    adv = load_json("advanced.json")
    names = (adv or {}).get("hmm", {}).get("state_names") or [f"State {i}" for i in range(len(d["posterior"][0]))]
    x = list(range(1, len(d["symbols"]) + 1))
    fig = go.Figure()
    for i in range(len(d["posterior"][0])):
        fig.add_trace(go.Scatter(x=x, y=[g[i] for g in d["posterior"]], mode="lines", name=names[i],
                                 line=dict(width=2, color=[C["s1"], C["s2"], C["s3"]][i % 3]),
                                 hovertemplate="window %{x}<br>P(" + names[i] + ")=%{y:.2f}<extra></extra>"))
    fig.update_layout(title="Hidden-state posterior under the scam HMM (Forward–Backward)", yaxis=dict(range=[0, 1.02], title="P(state)"),
                      xaxis_title="12-word window", hovermode="x unified")
    plot(fig, height=260, key="hmm_trace")
    sym = " → ".join(dict.fromkeys(s for s in d["symbols"] if s != "NEUTRAL")) or "only neutral windows"
    st.caption(f"Stage symbols in order of first appearance: {sym}. Log-likelihood ratio (scam HMM vs non-scam HMM) = {d['llr']:+.1f}. "
               "Supplementary view only — the HMM is not used for the prediction above.")


def render():
    st.title("Scam detection")
    st.markdown('<p class="lede">Paste the transcript of a call or chat after it happened. The analyzer estimates how likely it is to be a scam, '
                'shows which words drove that estimate, and flags known manipulation tactics separately.</p>', unsafe_allow_html=True)
    if not models_ready():
        missing_models_message()
        return
    az = get_analyzer()
    metrics = load_json("metrics.json") or {}

    st.session_state.setdefault("transcript", "")
    left, right = st.columns([3, 1.25], gap="large")
    with right:
        st.markdown("**Try an example**")
        ex = st.selectbox("Example transcript", list(EXAMPLES), label_visibility="collapsed", index=0)
        st.button("Load example", on_click=_set_text, args=(EXAMPLES[ex],), width="stretch")
        c1, c2 = st.columns(2)
        c1.button("Scam sample", on_click=_set_text, args=(EXAMPLES["Bank impersonation + OTP request (scam)"],), width="stretch")
        c2.button("Legit sample", on_click=_set_text, args=(EXAMPLES["Legitimate appointment reminder"],), width="stretch")
        st.markdown("**Model**")
        keys = az.available_models()
        best = metrics.get("best_model", az.best_model)
        labels = {k: MODEL_INFO[k]["name"] + ("  (default — best CV F1)" if k == best else "") for k in keys}
        model_key = st.selectbox("Model", keys, index=keys.index(best), format_func=labels.get, label_visibility="collapsed")
        compare = st.toggle("Also score with every model", value=False)
    with left:
        st.text_area("Conversation transcript", key="transcript", height=300,
                     placeholder="Caller: Hello, I'm calling from your bank's security team …\nYou: …")
        b1, b2, _ = st.columns([1.3, 1, 3])
        go_btn = b1.button("Analyze conversation", type="primary", width="stretch")
        b2.button("Clear", on_click=_set_text, args=("",), width="stretch")

    if go_btn:
        with st.spinner("Analysing …"):
            st.session_state["result"] = az.analyze(st.session_state["transcript"], model_key, compare_all=compare)
            st.session_state["result_text"] = st.session_state["transcript"]
    r = st.session_state.get("result")
    if not r:
        return
    if not r["ok"]:
        st.error(r["error"])
        return
    raw = st.session_state.get("result_text", "")

    st.divider()
    _verdict(r)

    t1, t2, t3, t4, t5 = st.tabs(["Why", "Red flags & advice", "Highlighted transcript", "Statistics", "Model details"])
    with t1:
        a, b = st.columns([1.15, 1], gap="large")
        with a:
            for line in r["explanation"]:
                st.markdown(line)
        with b:
            _evidence_chart(r["evidence"])
    with t2:
        a, b = st.columns([1.2, 1], gap="large")
        with a:
            st.markdown("#### Rule-based indicators")
            st.caption("Transparent phrase patterns, independent of the ML model. A match is a reason to look closer — not proof.")
            inds = r["indicators"]["indicators"]
            if not inds:
                st.markdown('<span class="chip ok">● No red-flag phrases found</span>', unsafe_allow_html=True)
            for d in inds:
                cls = "hi" if d["severity"] == "HIGH" else "md"
                icon = "▲" if d["severity"] == "HIGH" else "◆"
                chips = "".join(f'<span class="chip {cls}">{esc(p)}</span>' for p in d["phrases"])
                st.markdown(f'<div class="ind {cls}"><div class="t">{icon} {esc(d["label"])} '
                            f'<span style="color:var(--muted);font-weight:400">({d["severity"].lower()} severity)</span></div>'
                            f'<div class="w">{esc(d["why"])}</div>{chips}</div>', unsafe_allow_html=True)
            if r["indicators"]["safe_signals"]:
                st.markdown("**Reassuring phrases found**")
                st.markdown("".join(f'<span class="chip ok">{esc(s)}</span>' for s in r["indicators"]["safe_signals"]), unsafe_allow_html=True)
        with b:
            st.markdown("#### What to do")
            for tip in r["recommendations"]:
                st.markdown(f"- {tip}")
    with t3:
        st.markdown('<div class="legend-row"><mark class="r">rule red flag</mark> &nbsp; <mark class="m">term pushing toward scam</mark> '
                    '&nbsp; <mark class="g">term pushing toward legitimate</mark></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="transcript">{_highlight(raw, r)}</div>', unsafe_allow_html=True)
        with st.expander("What the model actually sees (after preprocessing)"):
            st.code(r["cleaned_preview"] + (" …" if len(r["cleaned_preview"].split()) >= 120 else ""), language=None)
        _hmm_trace(_cleaned_for(raw))
    with t4:
        s = r["stats"]
        cols = st.columns(4)
        cols[0].metric("Words", f"{s['words']:,}")
        cols[1].metric("Characters", f"{s['characters']:,}")
        cols[2].metric("Unique words", f"{s['unique_words']:,}")
        cols[3].metric("Turns / sentences", s["sentences_or_turns"])
        rows = [("Lexical diversity (unique ÷ total)", s["lexical_diversity"]), ("Average word length", s["avg_word_length"]),
                ("Questions (?)", s["questions"]), ("Exclamations (!)", s["exclamations"]), ("Uppercase letters share", f"{s['uppercase_ratio']:.1%}"),
                ("Numbers mentioned", s["numbers"]), ("Phone numbers", s["phone_numbers"]), ("Links", s["links"]),
                ("Money mentions", s["money_mentions"]), ("E-mail addresses", s["email_addresses"]),
                ("Terms known to the model", r["known_terms"]), ("Analysis time", f"{r['latency_ms']} ms")]
        st.markdown('<div class="panel kv">' + "".join(f"<div>{esc(k)}</div><div>{esc(v)}</div>" for k, v in rows) + "</div>",
                    unsafe_allow_html=True)
    with t5:
        mi = metrics.get("models", {}).get(r["model_key"], {})
        mt = mi.get("test", {})
        if mt:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("CV F1 (training)", f"{mi['cv']['f1']['mean']*100:.2f}%" if mi.get("cv") else "—")
            m2.metric("Test F1 (in-distribution)", f"{mt['main_test']['f1']*100:.2f}%")
            m3.metric("External test recall", f"{mt['ext_test']['recall']*100:.1f}%")
            m4.metric("Brier score (lower = better)", f"{mt['main_test'].get('brier', float('nan')):.4f}" if mt['main_test'].get('brier') is not None else "—")
        st.markdown(f"Prediction rule: **SCAM** if probability ≥ {r['thresholds']['decision']:.2f}. Risk: LOW < {r['thresholds']['medium']:.2f} ≤ MEDIUM < "
                    f"{r['thresholds']['high']:.2f} ≤ HIGH. Thresholds live in `config/settings.toml`.")
        if r["all_models"]:
            rows = sorted(r["all_models"].values(), key=lambda d: -d["score"])
            fig = go.Figure(go.Bar(x=[d["score"] for d in rows], y=[d["name"] for d in rows], orientation="h",
                                   marker=dict(color=[C["red"] if d["prediction"] else C["green"] for d in rows]),
                                   text=[f"{d['score']*100:.0f}%" for d in rows], textposition="outside",
                                   hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
            fig.add_vline(x=r["thresholds"]["decision"], line=dict(color=C["muted"], dash="dot"))
            fig.update_layout(title="Same transcript, every model (red = predicts scam)", xaxis=dict(range=[0, 1.1], title="P(scam) or vote share"),
                              yaxis=dict(automargin=True), showlegend=False)
            plot(fig, height=330, key="allmodels")
            st.caption("Hard voting shows the share of its three members voting scam (not a probability).")


def _cleaned_for(raw: str) -> str:
    from scam_analyzer.preprocessing import clean_text
    return clean_text(raw)
