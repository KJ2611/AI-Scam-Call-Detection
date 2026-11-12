"""Model-based evidence: exact per-term contributions for linear models.

For a linear model the scam log-odds (or SVM margin) is   b + Σ_i x_i · w_i
where x_i is the TF-IDF value of term i in THIS transcript and w_i the learned weight.
Each product x_i·w_i is therefore an exact, additive contribution — no approximation
(unlike LIME/SHAP for black-box models).

* Logistic Regression: w = coef_ (log-odds units).
* Calibrated Linear SVM: w = mean of the 5 fold-SVMs' coef_ (margin units; Platt scaling is a
  monotonic map applied afterwards, so the ranking of evidence is preserved).
* Multinomial NB: w_i = log P(term_i | scam) − log P(term_i | non-scam) (exact NB log-odds).
* Ensembles and tree models have no exact additive decomposition; the app shows Logistic
  Regression evidence for them and says so.
"""
from __future__ import annotations

import numpy as np

from .preprocessing import pretty_token


def linear_weights(key: str, model) -> tuple[np.ndarray, float, str] | None:
    if key == "logistic_regression":
        return model.coef_.ravel(), float(model.intercept_[0]), "log-odds"
    if key == "linear_svm_calibrated":
        ests = [cc.estimator for cc in model.calibrated_classifiers_]
        w = np.mean([e.coef_.ravel() for e in ests], axis=0)
        b = float(np.mean([e.intercept_[0] for e in ests]))
        return w, b, "SVM margin"
    if key == "naive_bayes":
        w = model.feature_log_prob_[1] - model.feature_log_prob_[0]
        b = float(model.class_log_prior_[1] - model.class_log_prior_[0])
        return w, b, "NB log-odds"
    return None


def contributions(x_row, weights: np.ndarray, feature_names: np.ndarray, top_k: int = 12) -> dict:
    """Split a single TF-IDF row into terms pushing toward SCAM and toward NON-SCAM."""
    x_row = x_row.tocsr()
    idx, vals = x_row.indices, x_row.data
    contrib = vals * weights[idx]
    order = np.argsort(contrib)
    to_scam = [(pretty_token(feature_names[idx[i]]), float(contrib[i]), float(vals[i]), float(weights[idx[i]]))
               for i in order[::-1] if contrib[i] > 0][:top_k]
    to_safe = [(pretty_token(feature_names[idx[i]]), float(contrib[i]), float(vals[i]), float(weights[idx[i]]))
               for i in order if contrib[i] < 0][:top_k]
    return {
        "toward_scam": to_scam,
        "toward_non_scam": to_safe,
        "sum_scam": float(contrib[contrib > 0].sum()),
        "sum_non_scam": float(contrib[contrib < 0].sum()),
        "n_terms": int(len(idx)),
    }


def global_top_terms(weights: np.ndarray, feature_names: np.ndarray, k: int = 25) -> dict:
    order = np.argsort(weights)
    return {
        "scam": [(pretty_token(feature_names[i]), float(weights[i])) for i in order[::-1][:k]],
        "non_scam": [(pretty_token(feature_names[i]), float(weights[i])) for i in order[:k]],
    }
