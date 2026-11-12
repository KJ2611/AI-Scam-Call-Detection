"""Model definitions. Every classifier consumes the same TF-IDF matrix.

Why each model is included (kept from the original notebooks, nothing added for show):
* Logistic Regression — linear, natively probabilistic, coefficients are directly explainable.
* Multinomial Naive Bayes — classic fast text baseline; probabilities are known to be over-confident.
* Linear SVM — strongest linear text classifier in the notebooks; has NO native probabilities, so it
  is wrapped in Platt (sigmoid) calibration via ``CalibratedClassifierCV`` (5-fold, training data only).
* Hard Voting (LR + NB + SVM) — majority vote; outputs a vote count, not a probability.
* Soft Voting (LR + NB + calibrated SVM) — averages probabilities. The original notebook had to drop
  the SVM because it has no ``predict_proba``; calibration lets it take part.
* Stacking (LR, NB, SVM → LR meta-learner, cv=5) — learns how to weight the base models.
* Bagging (50 decision trees) and AdaBoost (50 stumps) — tree ensembles, kept for comparison; they
  are slower and not linear, so they cannot give exact per-word explanations.
"""
from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, StackingClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_STATE

# Kept from the original notebooks; a 3-fold CV study on the training set (reports/tuning_study.json)
# found sublinear_tf/min_df/20k features changed F1 by <= 0.05 points, so the original setting is retained.
TFIDF_PARAMS = dict(max_features=10000, ngram_range=(1, 2))

MODEL_INFO = {
    "logistic_regression": {"name": "Logistic Regression", "family": "Linear", "probabilistic": True, "explainable": "exact"},
    "naive_bayes": {"name": "Multinomial Naive Bayes", "family": "Probabilistic", "probabilistic": True, "explainable": "exact"},
    "linear_svm_calibrated": {"name": "Linear SVM (Platt-calibrated)", "family": "Linear (margin)", "probabilistic": True, "explainable": "exact"},
    "hard_voting": {"name": "Hard Voting (LR+NB+SVM)", "family": "Ensemble", "probabilistic": False, "explainable": "via LR"},
    "soft_voting": {"name": "Soft Voting (LR+NB+cal. SVM)", "family": "Ensemble", "probabilistic": True, "explainable": "via LR"},
    "stacking": {"name": "Stacking (LR+NB+SVM → LR)", "family": "Ensemble", "probabilistic": True, "explainable": "via LR"},
    "bagging": {"name": "Bagging (50 Decision Trees)", "family": "Ensemble (trees)", "probabilistic": True, "explainable": "via LR"},
    "adaboost": {"name": "AdaBoost (50 stumps)", "family": "Ensemble (boosting)", "probabilistic": True, "explainable": "via LR"},
}


def make_vectorizer(**overrides) -> TfidfVectorizer:
    params = {**TFIDF_PARAMS, **overrides}
    return TfidfVectorizer(**params)


def _lr():
    # C=10 chosen by CV on training data (F1 0.9935 vs 0.9908 at the default C=1)
    return LogisticRegression(max_iter=2000, C=10.0, solver="liblinear")


def _svm():
    return LinearSVC(C=1.0, random_state=RANDOM_STATE)


def _cal_svm():
    return CalibratedClassifierCV(_svm(), method="sigmoid", cv=5)


def make_model(key: str, n_jobs: int = 1):
    if key == "logistic_regression":
        return _lr()
    if key == "naive_bayes":
        return MultinomialNB(alpha=0.1)
    if key == "linear_svm_calibrated":
        return _cal_svm()
    if key == "hard_voting":
        return VotingClassifier([("lr", _lr()), ("nb", MultinomialNB(alpha=0.1)), ("svm", _svm())], voting="hard")
    if key == "soft_voting":
        return VotingClassifier([("lr", _lr()), ("nb", MultinomialNB(alpha=0.1)), ("svm", _cal_svm())], voting="soft")
    if key == "stacking":
        return StackingClassifier(
            [("lr", _lr()), ("nb", MultinomialNB(alpha=0.1)), ("svm", _svm())],
            final_estimator=LogisticRegression(max_iter=2000), cv=5, n_jobs=n_jobs)
    if key == "bagging":
        return BaggingClassifier(DecisionTreeClassifier(random_state=RANDOM_STATE), n_estimators=50,
                                 max_samples=0.8, bootstrap=True, random_state=RANDOM_STATE, n_jobs=n_jobs)
    if key == "adaboost":
        return AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=RANDOM_STATE)
    raise KeyError(key)


def positive_scores(model, X):
    """Probability of SCAM for probabilistic models; vote fraction for hard voting."""
    if hasattr(model, "predict_proba") and not (isinstance(model, VotingClassifier) and model.voting == "hard"):
        return model.predict_proba(X)[:, 1]
    if isinstance(model, VotingClassifier):
        votes = [est.predict(X) for est in model.estimators_]
        return sum(votes) / len(votes)
    return model.decision_function(X)
