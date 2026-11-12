"""Exploratory data analysis → reports/eda.json (consumed by the Dataset Explorer page).

All statistics are computed from the actual processed data. Interpretations that were wrong in
the original eda.docx are corrected here (see docs/01_audit_report.md §7).
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from .config import DATA_RAW, REPORTS_DIR, ensure_dirs
from .data import load_split
from .preprocessing import PLACEHOLDERS

_MASK_PATTERNS = [
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "<EMAIL>"),
    (re.compile(r"(?:https?://|www\.)\S+"), "<URL>"),
    (re.compile(r"\+?\d[\d\s\-()]{7,}\d"), "<PHONE>"),
]


def mask_pii(text: str) -> str:
    for rx, rep in _MASK_PATTERNS:
        text = rx.sub(rep, text)
    return text


def _desc(s: pd.Series) -> dict:
    return {k: float(v) for k, v in {
        "mean": s.mean(), "median": s.median(), "p25": s.quantile(.25), "p75": s.quantile(.75),
        "p95": s.quantile(.95), "max": s.max(), "min": s.min()}.items()}


def _top_ngrams(texts, ngram, k=20):
    cv = CountVectorizer(ngram_range=(ngram, ngram), stop_words="english", min_df=3,
                         token_pattern=r"(?u)\b[a-z][a-z]+\b")
    X = cv.fit_transform(texts)
    doc_freq = np.asarray((X > 0).sum(axis=0)).ravel()
    names = cv.get_feature_names_out()
    order = np.argsort(doc_freq)[::-1][:k]
    return [(names[i], int(doc_freq[i])) for i in order]


def _distinctive(texts_a, texts_b, k=25):
    """Smoothed log-odds ratio of document frequency (a vs b), unigrams+bigrams, min support 30."""
    cv = CountVectorizer(ngram_range=(1, 2), stop_words="english", min_df=30, binary=True,
                         token_pattern=r"(?u)\b[a-z][a-z]+\b")
    cv.fit(list(texts_a) + list(texts_b))
    A = np.asarray(cv.transform(texts_a).sum(axis=0)).ravel()
    B = np.asarray(cv.transform(texts_b).sum(axis=0)).ravel()
    la = np.log((A + 1) / (len(texts_a) + 2)) - np.log((B + 1) / (len(texts_b) + 2))
    names = cv.get_feature_names_out()
    order = np.argsort(la)[::-1][:k]
    return [(names[i], float(la[i]), int(A[i]), int(B[i])) for i in order]


def main() -> dict:
    ensure_dirs()
    comp = pd.concat([load_split("main_train"), load_split("main_test")], ignore_index=True)
    eng = load_split("english_all")
    df = comp.copy()
    toks = df["text"].str.split()
    df["word_count"] = toks.str.len()
    df["char_count"] = df["text"].str.len()
    df["unique_words"] = toks.map(lambda t: len(set(t)))
    df["avg_word_len"] = df["char_count"] / df["word_count"].clip(lower=1)
    df["question_marks"] = df["raw_text"].astype(str).str.count(r"\?")
    df["placeholder_share"] = toks.map(lambda t: sum(w in PLACEHOLDERS for w in t) / max(len(t), 1))
    raw = df["raw_text"].astype(str)
    df["list_format"] = raw.str.contains(r"""^\s*['"].*['"]\s*,\s*['"]""", regex=True)

    out: dict = {}
    out["counts"] = {
        "composite_clean_rows": int(len(df)), "scam": int(df.label.sum()), "non_scam": int((df.label == 0).sum()),
        "scam_share": float(df.label.mean()), "english_scripts_rows": int(len(eng)),
        "english_scam": int(eng.label.sum()),
    }
    out["length"] = {lab: {c: _desc(df.loc[df.label == v, c]) for c in ("word_count", "char_count", "unique_words")}
                     for lab, v in (("scam", 1), ("non_scam", 0))}
    cap = float(df["word_count"].quantile(0.99))
    edges = np.linspace(0, cap, 41)
    out["word_hist"] = {"edges": edges.tolist(), "cap_p99": cap,
                        "scam": np.histogram(df.loc[df.label == 1, "word_count"].clip(upper=cap), edges)[0].tolist(),
                        "non_scam": np.histogram(df.loc[df.label == 0, "word_count"].clip(upper=cap), edges)[0].tolist()}
    log_edges = np.logspace(0, np.log10(df["word_count"].max() + 1), 41)
    out["word_hist_log"] = {"edges": log_edges.tolist(),
                            "scam": np.histogram(df.loc[df.label == 1, "word_count"], log_edges)[0].tolist(),
                            "non_scam": np.histogram(df.loc[df.label == 0, "word_count"], log_edges)[0].tolist()}
    cols = ["label", "word_count", "char_count", "unique_words", "avg_word_len", "question_marks", "placeholder_share"]
    out["corr"] = {"columns": cols,
                   "pearson": df[cols].corr("pearson").round(3).values.tolist(),
                   "spearman": df[cols].corr("spearman").round(3).values.tolist()}
    out["format"] = {"list_format_scam": int(df.loc[df.label == 1, "list_format"].sum()),
                     "list_format_non_scam": int(df.loc[df.label == 0, "list_format"].sum())}
    # vocabulary
    voc = {}
    for lab, v in (("scam", 1), ("non_scam", 0)):
        words = [w for t in toks[df.label == v] for w in t]
        vc = pd.Series(words).value_counts()
        voc[lab] = {"tokens": int(len(words)), "vocabulary": int(len(vc)), "type_token_ratio": float(len(vc) / max(len(words), 1)),
                    "hapax": int((vc == 1).sum())}
    out["vocabulary"] = voc
    scam_t, non_t = df.loc[df.label == 1, "text"], df.loc[df.label == 0, "text"]
    out["top_unigrams"] = {"scam": _top_ngrams(scam_t, 1, 25), "non_scam": _top_ngrams(non_t, 1, 25)}
    out["top_bigrams"] = {"scam": _top_ngrams(scam_t, 2, 20), "non_scam": _top_ngrams(non_t, 2, 20)}
    out["distinctive"] = {"scam": _distinctive(scam_t, non_t, 25), "non_scam": _distinctive(non_t, scam_t, 25)}
    # source-artifact tokens counted on RAW composite text (before masking)
    rawc = pd.read_csv(DATA_RAW / "composite_train.csv")
    low = rawc["text"].astype(str).str.lower()
    art = {}
    for w in ["zhang", "xiao", "li", "rupees", "wechat", "alipay", "otp", "kyc", "upi", "aadhaar", "anydesk"]:
        m = low.str.contains(rf"\b{w}\b", regex=True)
        art[w] = {"scam": int(m[rawc.label == 1].sum()), "non_scam": int(m[rawc.label == 0].sum())}
    out["artifact_tokens"] = art
    # samples (PII-masked, truncated)
    rng = np.random.RandomState(7)
    samp = []
    for v in (1, 0):
        sub = df[(df.label == v) & (df.word_count.between(25, 400))]
        for i in rng.choice(sub.index, 8, replace=False):
            r = sub.loc[i]
            samp.append({"label": int(v), "words": int(r.word_count), "text": mask_pii(str(r.raw_text))[:500]})
    for i in rng.choice(eng.index, 4, replace=False):
        samp.append({"label": int(eng.loc[i, "label"]), "words": len(eng.loc[i, "text"].split()),
                     "text": mask_pii(str(eng.loc[i, "raw_text"]))[:500], "source": "english_scripts"})
    out["samples"] = samp
    (REPORTS_DIR / "eda.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    o = main()
    print(json.dumps({k: o[k] for k in ("counts", "format", "vocabulary")}, indent=1))
