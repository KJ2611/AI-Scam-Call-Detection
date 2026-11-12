"""Supplementary unsupervised & sequential analysis → reports/advanced.json + models/hmm_*.joblib

Fixes relative to the original Clustering_and_HMM notebook (audit §6):
* All clustering methods run in ONE feature space (50-D LSA = TF-IDF → TruncatedSVD → L2-norm),
  and every metric is computed in that same space. 2-D coordinates are for plotting only.
* Cluster IDs are never treated as labels; agreement with labels is measured with ARI / NMI / purity.
* HMM observations are interpretable "stage symbols" (AUTHORITY, URGENCY, CREDENTIAL …) on fixed
  12-token windows, so no word is silently dropped and the sequence does not depend on formatting.
* One HMM per class (scam / non-scam) trained with Baum–Welch on TRAINING data; classification by
  log-likelihood ratio on TEST data — this is what tells us whether sequence modelling helps.
* Our from-scratch scaled Forward/Backward/Viterbi are verified against hmmlearn.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
from hmmlearn.hmm import CategoricalHMM
from sklearn.cluster import AgglomerativeClustering, Birch, KMeans, SpectralClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             f1_score, normalized_mutual_info_score, precision_score, recall_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Normalizer

from .config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR, ensure_dirs
from .data import load_split
from .hmm_algorithms import backward, forward, posteriors, unscaled_forward_probability, viterbi
from .indicators import HMM_SYMBOLS, text_to_symbols

RNG = np.random.RandomState(RANDOM_STATE)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ============================================================================ clustering
def clustering(n_sample: int = 4000) -> dict:
    df = load_split("train_full")
    idx = []
    for lab in (0, 1):   # stratified sample
        pool = np.where(df.label.to_numpy() == lab)[0]
        idx.extend(RNG.choice(pool, int(n_sample * (pool.size / len(df))), replace=False))
    idx = np.array(sorted(idx))
    texts = np.array(df.text.tolist(), dtype=object)[idx]
    y = df.label.to_numpy()[idx]

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    T = tfidf.fit_transform(texts)
    svd = TruncatedSVD(50, random_state=RANDOM_STATE)
    Z = Normalizer().fit_transform(svd.fit_transform(T))
    xy = TruncatedSVD(2, random_state=RANDOM_STATE).fit_transform(T)

    methods = {
        "K-Means": lambda: KMeans(2, n_init=10, random_state=RANDOM_STATE).fit_predict(Z),
        "Gaussian Mixture (GMM)": lambda: GaussianMixture(2, covariance_type="diag", random_state=RANDOM_STATE).fit(Z).predict(Z),
        "Agglomerative – Ward": lambda: AgglomerativeClustering(2, linkage="ward").fit_predict(Z),
        "Agglomerative – Single": lambda: AgglomerativeClustering(2, linkage="single").fit_predict(Z),
        "Agglomerative – Complete": lambda: AgglomerativeClustering(2, linkage="complete").fit_predict(Z),
        "Agglomerative – Average": lambda: AgglomerativeClustering(2, linkage="average").fit_predict(Z),
        "Spectral (k-NN graph)": lambda: SpectralClustering(2, affinity="nearest_neighbors", n_neighbors=15,
                                                            random_state=RANDOM_STATE).fit_predict(Z),
        "BIRCH": lambda: Birch(threshold=0.3, n_clusters=2).fit_predict(Z),
    }
    rows, assignments = [], {}
    for name, fn in methods.items():
        t = time.time()
        lab = np.asarray(fn())
        k = len(np.unique(lab))
        sizes = np.bincount(lab).tolist()
        purity = sum(np.bincount(y[lab == c], minlength=2).max() for c in np.unique(lab)) / len(y)
        scam_share = [float(y[lab == c].mean()) for c in range(k)]
        degenerate = min(sizes) < 0.01 * len(y)
        rows.append({
            "method": name, "clusters": k, "sizes": sizes, "scam_share_per_cluster": scam_share,
            "silhouette": float(silhouette_score(Z, lab)) if k > 1 else None,
            "davies_bouldin": float(davies_bouldin_score(Z, lab)) if k > 1 else None,
            "calinski_harabasz": float(calinski_harabasz_score(Z, lab)) if k > 1 else None,
            "ari": float(adjusted_rand_score(y, lab)), "nmi": float(normalized_mutual_info_score(y, lab)),
            "purity": float(purity), "degenerate": bool(degenerate), "seconds": round(time.time() - t, 1),
        })
        assignments[name] = lab
        log(f"  {name}: sil={rows[-1]['silhouette']:.3f} ARI={rows[-1]['ari']:.3f} sizes={sizes}")

    # choosing k for K-Means (silhouette / inertia curves)
    k_curve = []
    for k in range(2, 11):
        km = KMeans(k, n_init=10, random_state=RANDOM_STATE).fit(Z)
        k_curve.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(silhouette_score(Z, km.labels_))})
    # topic view: K-Means with k=8, top terms + scam share per cluster
    k_topics = 8
    km = KMeans(k_topics, n_init=10, random_state=RANDOM_STATE).fit(Z)
    centers_tfidf = svd.inverse_transform(km.cluster_centers_)
    terms = tfidf.get_feature_names_out()
    topics = []
    for c in range(k_topics):
        m = km.labels_ == c
        topics.append({"cluster": c, "size": int(m.sum()), "scam_share": float(y[m].mean()),
                       "top_terms": [terms[i] for i in np.argsort(centers_tfidf[c])[::-1][:10]]})
    topics.sort(key=lambda d: -d["scam_share"])

    plot_idx = RNG.choice(len(y), min(1500, len(y)), replace=False)
    return {
        "sample_size": int(len(y)), "majority_share": float(max(y.mean(), 1 - y.mean())), "feature_space": "TF-IDF (5k, 1-2 grams, English stop-words) → TruncatedSVD(50) → L2-normalised",
        "results": rows, "k_curve": k_curve, "kmeans_topics": topics,
        "plot": {"x": xy[plot_idx, 0].round(4).tolist(), "y": xy[plot_idx, 1].round(4).tolist(),
                 "label": y[plot_idx].tolist(),
                 "clusters": {n: a[plot_idx].tolist() for n, a in assignments.items()},
                 "kmeans8": km.labels_[plot_idx].tolist()},
        "original_notebook": {
            "issues": ["K-Means fitted on 5,000-D TF-IDF but scored in 2-D SVD space with the other methods",
                       "Conclusion text named K-Means as best ARI (0.0578) although Ward had 0.1309",
                       "Single linkage produced a 4,999 / 1 split (degenerate) but was still ranked",
                       "DBSCAN listed in the introduction but never implemented"],
            "reported_silhouette_gmm": 0.5831, "reported_best_ari_ward": 0.1309},
    }


# ============================================================================ HMM
def _seqs(df, max_n=None, min_len=3):
    out, labels = [], []
    rows = df.sample(n=min(max_n, len(df)), random_state=RANDOM_STATE) if max_n else df
    for t, l in zip(rows.text, rows.label):
        s = text_to_symbols(t)
        if len(s) >= min_len:
            out.append(np.array(s)); labels.append(int(l))
    return out, np.array(labels)


def _fit_hmm(seqs, n_states=3, seed=RANDOM_STATE):
    X = np.concatenate(seqs).reshape(-1, 1)
    lengths = [len(s) for s in seqs]
    best = None
    for s in range(3):  # a few random restarts; keep the best training likelihood
        m = CategoricalHMM(n_components=n_states, n_iter=200, tol=1e-4, random_state=seed + s,
                           init_params="ste", n_features=len(HMM_SYMBOLS))
        m.fit(X, lengths)
        ll = m.score(X, lengths)
        if best is None or ll > best[1]:
            best = (m, ll)
    return best[0]


def _bag_of_symbols(seqs):
    return np.array([np.bincount(s, minlength=len(HMM_SYMBOLS)) for s in seqs])


def hmm_analysis(n_train_per_class: int = 4000, n_states: int = 3) -> dict:
    tr = load_split("train_full")
    s_scam, _ = _seqs(tr[tr.label == 1], n_train_per_class)
    s_non, _ = _seqs(tr[tr.label == 0], n_train_per_class)
    log(f"  HMM training sequences: scam={len(s_scam)} non-scam={len(s_non)}")
    hmm_s = _fit_hmm(s_scam, n_states)
    hmm_n = _fit_hmm(s_non, n_states)
    prior = np.log(len(s_scam) / (len(s_scam) + len(s_non)))
    prior_n = np.log(len(s_non) / (len(s_scam) + len(s_non)))
    joblib.dump({"scam": hmm_s, "non_scam": hmm_n, "symbols": HMM_SYMBOLS, "log_prior": (prior_n, prior)},
                MODELS_DIR / "hmm_models.joblib", compress=3)

    # baseline that ignores ORDER: multinomial NB on symbol counts
    nb = MultinomialNB().fit(_bag_of_symbols(s_scam + s_non), np.r_[np.ones(len(s_scam)), np.zeros(len(s_non))])

    evals = {}
    for split in ("main_test", "ext_test"):
        df = load_split(split)
        seqs, y = _seqs(df, 3000 if split == "main_test" else None, min_len=1)
        llr = np.array([hmm_s.score(s.reshape(-1, 1)) - hmm_n.score(s.reshape(-1, 1)) for s in seqs]) + (prior - prior_n)
        p_hmm = (llr > 0).astype(int)
        p_bag = nb.predict(_bag_of_symbols(seqs)).astype(int)
        evals[split] = {
            "n": int(len(y)),
            "hmm": {"accuracy": float(accuracy_score(y, p_hmm)), "f1": float(f1_score(y, p_hmm)),
                    "precision": float(precision_score(y, p_hmm, zero_division=0)), "recall": float(recall_score(y, p_hmm))},
            "bag_of_symbols_nb": {"accuracy": float(accuracy_score(y, p_bag)), "f1": float(f1_score(y, p_bag)),
                                  "precision": float(precision_score(y, p_bag, zero_division=0)), "recall": float(recall_score(y, p_bag))},
            "majority_baseline": float(max(y.mean(), 1 - y.mean())),
        }
        log(f"  {split}: HMM acc={evals[split]['hmm']['accuracy']:.3f}  bag-of-symbols acc={evals[split]['bag_of_symbols_nb']['accuracy']:.3f}")

    # verification of the from-scratch algorithms against hmmlearn
    checks = []
    for s in s_scam[:25]:
        pi, A, B = hmm_s.startprob_, hmm_s.transmat_, hmm_s.emissionprob_
        _, c, ll = forward(s, pi, A, B)
        path, vlp = viterbi(s, pi, A, B)
        ref_vlp, ref_path = hmm_s.decode(s.reshape(-1, 1), algorithm="viterbi")
        checks.append({"len": int(len(s)), "ll_ours": ll, "ll_hmmlearn": float(hmm_s.score(s.reshape(-1, 1))),
                       "viterbi_same_path": bool(np.array_equal(path, ref_path)), "viterbi_lp_diff": abs(vlp - ref_vlp)})
    long_seq = np.resize(np.concatenate(s_scam[:50]), 1200)   # T = 1,200 windows
    underflow = {"length": int(len(long_seq)),
                 "unscaled_probability": unscaled_forward_probability(long_seq, hmm_s.startprob_, hmm_s.transmat_, hmm_s.emissionprob_),
                 "scaled_log_likelihood": forward(long_seq, hmm_s.startprob_, hmm_s.transmat_, hmm_s.emissionprob_)[2]}

    def params(m):
        return {"startprob": m.startprob_.round(4).tolist(), "transmat": m.transmat_.round(4).tolist(),
                "emissionprob": m.emissionprob_.round(4).tolist()}

    # symbol frequency by class (what the HMM sees)
    freq = {"scam": (np.bincount(np.concatenate(s_scam), minlength=len(HMM_SYMBOLS)) / sum(map(len, s_scam))).round(4).tolist(),
            "non_scam": (np.bincount(np.concatenate(s_non), minlength=len(HMM_SYMBOLS)) / sum(map(len, s_non))).round(4).tolist()}
    def state_names(m):
        return hmm_state_names(m)

    return {
        "symbols": HMM_SYMBOLS, "window_tokens": 12, "n_states": n_states,
        "state_names": state_names(hmm_s), "state_names_non": state_names(hmm_n),
        "train_sequences": {"scam": len(s_scam), "non_scam": len(s_non)},
        "symbol_frequency": freq,
        "scam_hmm": params(hmm_s), "non_scam_hmm": params(hmm_n),
        "evaluation": evals,
        "verification": {"max_abs_loglik_diff": float(max(abs(c["ll_ours"] - c["ll_hmmlearn"]) for c in checks)),
                         "viterbi_paths_identical": int(sum(c["viterbi_same_path"] for c in checks)), "n_checked": len(checks),
                         "underflow_demo": underflow},
        "original_notebook": {
            "issues": ["Vocabulary limited to the 100 most frequent words (mostly stop-words); all other words deleted from sequences",
                       "A single HMM trained on mixed scam + non-scam text cannot discriminate between them",
                       "Unscaled forward pass (1e-226 at T=121) underflows for longer sequences",
                       "'Baum–Welch' section re-fitted the identical model (duplicate output)",
                       "Gaussian HMM fitted on 30,409 independent conversations treated as one time series"]},
    }


def hmm_state_names(m) -> list[str]:
    """Readable state names: top non-neutral emissions + share of NEUTRAL windows."""
    names = []
    neu = HMM_SYMBOLS.index("NEUTRAL")
    for row in m.emissionprob_:
        order = [i for i in np.argsort(row)[::-1] if i != neu and row[i] >= 0.05][:2]
        top = " / ".join(HMM_SYMBOLS[i].title().replace("_", " ") for i in order) or "Neutral talk"
        names.append(f"{top} ({row[neu]:.0%} neutral)")
    return [f"{n} #{i}" if names.count(n) > 1 else n for i, n in enumerate(names)]


def decode_transcript(cleaned_text: str, bundle=None) -> dict | None:
    """Live decoding used by the app: symbols, Viterbi path and state posteriors under the scam HMM."""
    bundle = bundle or joblib.load(MODELS_DIR / "hmm_models.joblib")
    seq = np.array(text_to_symbols(cleaned_text))
    if len(seq) == 0:
        return None
    m_s, m_n = bundle["scam"], bundle["non_scam"]
    pi, A, B = m_s.startprob_, m_s.transmat_, m_s.emissionprob_
    alpha, c, ll_s = forward(seq, pi, A, B)
    beta = backward(seq, A, B, c)
    gamma = posteriors(alpha, beta)
    path, _ = viterbi(seq, pi, A, B)
    ll_n = forward(seq, m_n.startprob_, m_n.transmat_, m_n.emissionprob_)[2]
    return {"symbols": [HMM_SYMBOLS[i] for i in seq], "viterbi": path.tolist(), "posterior": gamma.round(4).tolist(),
            "loglik_scam": ll_s, "loglik_non_scam": ll_n, "llr": ll_s - ll_n + (bundle["log_prior"][1] - bundle["log_prior"][0])}


def main():
    ensure_dirs()
    t = time.time()
    log("Clustering …")
    cl = clustering()
    log("HMM …")
    hm = hmm_analysis()
    out = {"clustering": cl, "hmm": hm, "seconds": round(time.time() - t, 1)}
    (REPORTS_DIR / "advanced.json").write_text(json.dumps(out, indent=1))
    log(f"saved reports/advanced.json ({out['seconds']}s)")
    return out


if __name__ == "__main__":
    main()
