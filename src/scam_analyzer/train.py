"""Train, cross-validate, evaluate and save every model.

Run from the project root:   python -m scam_analyzer.train        (with src/ on PYTHONPATH)
or simply:                    python scripts/train_models.py

Protocol (leakage-safe):
1. Clean + de-duplicate each source; split composite 80:20 (stratified, rs=42); split the external
   English scripts 50:50 and add one half to training.
2. Model selection = 5-fold stratified CV on the TRAINING split only, with TF-IDF re-fitted inside
   every fold (sklearn Pipeline). The test sets are not used to choose anything.
3. One TF-IDF vectorizer is fitted on the full training split; every model is fitted on it.
4. Each model is evaluated ONCE on: main_test (in-distribution), ext_test (external source),
   composite_test_unseen and fraud_call_unseen.
5. Ablations explain *why* scores look the way they do (historical reproduction, out-of-domain,
   artifact masking, format/length shortcut, near-duplicate filtering).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, log_loss, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from . import __version__
from .config import DATA_RAW, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, ensure_dirs
from .data import build_datasets
from .explain import global_top_terms, linear_weights
from .models import MODEL_INFO, TFIDF_PARAMS, make_model, make_vectorizer, positive_scores
from .preprocessing import clean_many, legacy_clean_text

EVAL_SETS = {
    "main_test": "In-distribution test (composite, 20%)",
    "ext_test": "External test (English scripts, held-out 50%)",
    "composite_test_unseen": "composite_test.csv (rows not in training file)",
    "fraud_call_unseen": "fraud_call.file (rows not in training file)",
}


def _arr(s: pd.Series) -> np.ndarray:
    return np.array(s.tolist(), dtype=object)


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def ece(y, p, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def reliability(y, p, n_bins: int = 10) -> list[dict]:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out.append({"bin_mid": float((bins[b] + bins[b + 1]) / 2), "mean_pred": float(p[m].mean()),
                        "frac_pos": float(y[m].mean()), "n": int(m.sum())})
    return out


def _downsample_curve(xs, ys, n=120):
    xs, ys = np.asarray(xs), np.asarray(ys)
    if len(xs) <= n:
        return xs.tolist(), ys.tolist()
    keep = np.unique(np.linspace(0, len(xs) - 1, n).astype(int))
    return xs[keep].tolist(), ys[keep].tolist()


def evaluate(y, pred, score=None, probabilistic=True) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    r = {
        "n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "false_negatives": int(fn), "false_positives": int(fp),
    }
    if score is not None and len(np.unique(y)) == 2:
        r["roc_auc"] = float(roc_auc_score(y, score))
        r["average_precision"] = float(average_precision_score(y, score))
        if probabilistic:
            p = np.clip(score, 1e-6, 1 - 1e-6)
            r["brier"] = float(brier_score_loss(y, p))
            r["log_loss"] = float(log_loss(y, p))
            r["ece"] = ece(y, p)
            r["reliability"] = reliability(y, p)
        fpr, tpr, _ = roc_curve(y, score)
        r["roc_curve"] = dict(zip(("fpr", "tpr"), _downsample_curve(fpr, tpr)))
        prec, rec, _ = precision_recall_curve(y, score)
        r["pr_curve"] = dict(zip(("recall", "precision"), _downsample_curve(rec, prec)))
    return r


# ----------------------------------------------------------------------------- ablations
def ablation_historical() -> dict:
    """Re-run the ORIGINAL notebook protocol (legacy cleaning, composite only, default models)."""
    tr = pd.read_csv(DATA_RAW / "composite_train.csv")
    tr["c"] = tr["text"].map(legacy_clean_text)
    tr = tr[tr["c"].str.len() > 0].drop_duplicates(["c", "label"]).reset_index(drop=True)
    Xa, Xb, ya, yb = train_test_split(_arr(tr["c"]), tr["label"].to_numpy(), test_size=0.2,
                                      random_state=42, stratify=tr["label"])
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    A, B = v.fit_transform(Xa), v.transform(Xb)
    out = {"rows": int(len(tr)), "scam": int(tr["label"].sum()), "models": {}}
    reported = {"Logistic Regression": 0.9887, "Naive Bayes": 0.9507, "Linear SVM": 0.9942}
    for name, m in [("Logistic Regression", LogisticRegression(max_iter=1000)), ("Naive Bayes", MultinomialNB()),
                    ("Linear SVM", LinearSVC())]:
        p = m.fit(A, ya).predict(B)
        out["models"][name] = {"reported_accuracy": reported[name], "reproduced_accuracy": float(accuracy_score(yb, p)),
                               "reproduced_f1": float(f1_score(yb, p))}
    return out


def ablation_out_of_domain(splits) -> dict:
    """Train on composite ONLY, test on ALL English scripts (never seen)."""
    v = make_vectorizer()
    A = v.fit_transform(_arr(splits["main_train"]["text"]))
    y = splits["main_train"]["label"].to_numpy()
    E = v.transform(_arr(splits["english_all"]["text"]))
    ye = splits["english_all"]["label"].to_numpy()
    out = {}
    for key in ("logistic_regression", "linear_svm_calibrated", "naive_bayes"):
        m = make_model(key).fit(A, y)
        s = positive_scores(m, E)
        out[key] = evaluate(ye, (s >= 0.5).astype(int), s)
        for k in ("roc_curve", "pr_curve", "reliability"):
            out[key].pop(k, None)
    return out


def ablation_masking(splits) -> dict:
    """Effect of removing source-artifact tokens (LR, same split)."""
    out = {}
    for masked in (True, False):
        def prep(df):
            return _arr(df["text"]) if masked else np.array(clean_many(df["raw_text"], mask_source_artifacts=False), dtype=object)
        v = make_vectorizer()
        A = v.fit_transform(prep(splits["train_full"]))
        m = make_model("logistic_regression").fit(A, splits["train_full"]["label"])
        res = {}
        for s in ("main_test", "ext_test"):
            yy = splits[s]["label"].to_numpy()
            p = m.predict(v.transform(prep(splits[s])))
            res[s] = {"accuracy": float(accuracy_score(yy, p)), "f1": float(f1_score(yy, p)), "recall": float(recall_score(yy, p))}
        out["masked" if masked else "unmasked"] = res
    return out


def ablation_format_shortcut(splits) -> dict:
    """Can the label be predicted WITHOUT reading any words? (raw format + length only)"""
    def feats(raw: pd.Series):
        raw = raw.astype(str)
        return np.c_[raw.str.len(), raw.str.split().str.len(),
                     raw.str.count(r"', '") + raw.str.count(r'", "'),
                     raw.str.startswith(("'", '"')).astype(int), raw.str.count(r"\?")]
    tr, te = splits["main_train"], splits["main_test"]
    rf = RandomForestClassifier(200, random_state=RANDOM_STATE, n_jobs=-1).fit(feats(tr["raw_text"]), tr["label"])
    p = rf.predict(feats(te["raw_text"]))
    wc = te["raw_text"].astype(str).str.split().str.len()
    fmt = te["raw_text"].astype(str).str.contains(r"""['"]\s*,\s*['"]""", regex=True)
    return {"format_length_only_rf_accuracy": float(accuracy_score(te["label"], p)),
            "format_length_only_rf_f1": float(f1_score(te["label"], p)),
            "majority_baseline_accuracy": float(max(te["label"].mean(), 1 - te["label"].mean())),
            "list_format_share": {"scam": float(fmt[te["label"] == 1].mean()), "non_scam": float(fmt[te["label"] == 0].mean())},
            "median_words": {"scam": float(wc[te["label"] == 1].median()), "non_scam": float(wc[te["label"] == 0].median())}}


def near_duplicate_mask(A, B, thr=0.9):
    nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(A)
    d, _ = nn.kneighbors(B)
    sim = 1 - d.ravel()
    return sim < thr, sim


# ----------------------------------------------------------------------------- main
def main(cv_folds: int = 5, skip_slow_cv: bool = False) -> dict:
    ensure_dirs()
    t0 = time.time()
    log("Building datasets …")
    splits, data_report = build_datasets(save=True)
    tr = splits["train_full"]
    X_text, y = _arr(tr["text"]), tr["label"].to_numpy()

    log("Fitting TF-IDF on the training split …")
    vec = make_vectorizer().fit(X_text)
    if hasattr(vec, "stop_words_"):
        vec.stop_words_ = None          # huge and not needed for transform()
    Xtr = vec.transform(X_text)
    feats = vec.get_feature_names_out()
    X_eval = {k: vec.transform(_arr(splits[k]["text"])) for k in EVAL_SETS}
    y_eval = {k: splits[k]["label"].to_numpy() for k in EVAL_SETS}
    joblib.dump(vec, MODELS_DIR / "tfidf_vectorizer.joblib", compress=3)

    nd_mask, nd_sim = near_duplicate_mask(Xtr, X_eval["main_test"], 0.9)

    cv = StratifiedKFold(cv_folds, shuffle=True, random_state=RANDOM_STATE)
    results: dict = {}
    for key, info in MODEL_INFO.items():
        log(f"== {info['name']}")
        res = {"info": info}
        # ---- cross-validation on the training split only (TF-IDF refit per fold)
        if skip_slow_cv and key in ("bagging", "adaboost"):
            res["cv"] = None
        else:
            t = time.time()
            pipe = make_pipeline(make_vectorizer(), make_model(key, n_jobs=1))
            cvr = cross_validate(pipe, X_text, y, cv=cv, scoring=["accuracy", "precision", "recall", "f1"], n_jobs=2)
            res["cv"] = {m: {"mean": float(cvr[f"test_{m}"].mean()), "std": float(cvr[f"test_{m}"].std())}
                         for m in ("accuracy", "precision", "recall", "f1")}
            res["cv"]["seconds"] = round(time.time() - t, 1)
            log(f"   CV F1 = {res['cv']['f1']['mean']:.4f} ± {res['cv']['f1']['std']:.4f}  ({res['cv']['seconds']}s)")
        # ---- final fit on the full training split
        t = time.time()
        model = make_model(key, n_jobs=2).fit(Xtr, y)
        res["fit_seconds"] = round(time.time() - t, 1)
        joblib.dump(model, MODELS_DIR / f"{key}.joblib", compress=3)
        res["size_kb"] = round((MODELS_DIR / f"{key}.joblib").stat().st_size / 1024, 1)
        # ---- evaluation on every held-out set
        res["test"] = {}
        for s in EVAL_SETS:
            score = positive_scores(model, X_eval[s])
            pred = model.predict(X_eval[s]) if key == "hard_voting" else (score >= 0.5).astype(int)
            res["test"][s] = evaluate(y_eval[s], pred, score, probabilistic=info["probabilistic"])
        pred_main = (positive_scores(model, X_eval["main_test"]) >= 0.5).astype(int) if key != "hard_voting" \
            else model.predict(X_eval["main_test"])
        res["test"]["main_test_no_near_duplicates"] = evaluate(y_eval["main_test"][nd_mask], pred_main[nd_mask])
        lw = linear_weights(key, model)
        if lw is not None:
            res["top_terms"] = global_top_terms(lw[0], feats, 25)
        log(f"   main_test acc={res['test']['main_test']['accuracy']:.4f} f1={res['test']['main_test']['f1']:.4f} | "
            f"ext_test acc={res['test']['ext_test']['accuracy']:.4f} recall={res['test']['ext_test']['recall']:.4f}")
        results[key] = res

    # ---- model selection: best CV F1 among probabilistic models (test sets NOT used)
    cand = {k: r["cv"]["f1"]["mean"] for k, r in results.items() if r["cv"] and r["info"]["probabilistic"]}
    best = max(cand, key=cand.get)
    log(f"Selected default model by CV F1: {best}")

    log("Ablations …")
    abl = {
        "historical_reproduction": ablation_historical(),
        "out_of_domain_english": ablation_out_of_domain(splits),
        "artifact_masking": ablation_masking(splits),
        "format_length_shortcut": ablation_format_shortcut(splits),
        "near_duplicates": {"threshold": 0.9, "share_of_main_test_with_train_neighbour": float((~nd_mask).mean()),
                            "share_ge_095": float((nd_sim >= 0.95).mean())},
    }

    metrics = {
        "version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tfidf": {k: (list(v) if isinstance(v, tuple) else v) for k, v in TFIDF_PARAMS.items()},
        "vocabulary_size": int(len(feats)),
        "cv_folds": cv_folds,
        "selection_metric": "Mean 5-fold CV F1 (scam class) on the training split, probabilistic models only",
        "best_model": best,
        "eval_sets": EVAL_SETS,
        "data": data_report,
        "models": results,
        "ablations": abl,
        "total_seconds": round(time.time() - t0, 1),
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=1))
    joblib.dump({"best_model": best, "feature_names": feats}, MODELS_DIR / "meta.joblib", compress=3)
    log(f"Done in {metrics['total_seconds']}s → reports/metrics.json, models/*.joblib")
    return metrics


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fast", action="store_true", help="skip CV for bagging/adaboost")
    a = ap.parse_args()
    main(a.folds, a.fast)
