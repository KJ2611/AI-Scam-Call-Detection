"""3-fold CV study of TF-IDF and model hyper-parameters on the TRAINING split only.

Run:  python scripts/tuning_study.py      (≈ 7 minutes on 2 CPU cores)
Writes reports/tuning_study.json, which the Model Comparison page displays.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_validate  # noqa: E402
from sklearn.naive_bayes import MultinomialNB  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from scam_analyzer.config import REPORTS_DIR  # noqa: E402
from scam_analyzer.data import load_split  # noqa: E402

tr = load_split("train_full")
X, y = np.array(tr.text.tolist(), dtype=object), tr.label.to_numpy()
cv = StratifiedKFold(3, shuffle=True, random_state=42)
configs = {
    "10k features, 1-2 grams (original)": dict(max_features=10000, ngram_range=(1, 2)),
    "10k, 1-2 grams, sublinear tf, min_df=2": dict(max_features=10000, ngram_range=(1, 2), sublinear_tf=True, min_df=2),
    "20k, 1-2 grams, sublinear tf, min_df=2": dict(max_features=20000, ngram_range=(1, 2), sublinear_tf=True, min_df=2),
    "10k features, unigrams only": dict(max_features=10000, ngram_range=(1, 1)),
}
models = {
    "Linear SVM (C=1)": LinearSVC(),
    "Logistic Regression (C=1)": LogisticRegression(max_iter=2000),
    "Logistic Regression (C=10)": LogisticRegression(max_iter=2000, C=10, solver="liblinear"),
    "Naive Bayes (alpha=1)": MultinomialNB(),
    "Naive Bayes (alpha=0.1)": MultinomialNB(alpha=0.1),
}
rows = []
for cname, cfg in configs.items():
    for mname, m in models.items():
        t = time.time()
        r = cross_validate(make_pipeline(TfidfVectorizer(**cfg), m), X, y, cv=cv, scoring=["f1", "recall", "precision"], n_jobs=2)
        rows.append({"tfidf": cname, "model": mname, "cv_f1": float(r["test_f1"].mean()),
                     "cv_recall": float(r["test_recall"].mean()), "cv_precision": float(r["test_precision"].mean()),
                     "seconds": round(time.time() - t, 1)})
        print(rows[-1], flush=True)
(REPORTS_DIR / "tuning_study.json").write_text(json.dumps({"folds": 3, "data": "train_full", "rows": rows}, indent=1))
print("saved", REPORTS_DIR / "tuning_study.json")
