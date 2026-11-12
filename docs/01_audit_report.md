# Phase 1 — Project Audit Report

**Project:** AI-Based Scam Conversation Risk Analyzer
**Audit scope:** 3 notebooks, `eda.docx`, two presentation PDFs, `project(2).zip` (5 data files).
**GitHub repository:** not provided (the prompt still contains the placeholder URL) and no scam-related repository was visible from the connected GitHub account. Everything below is based on the uploaded files only.

All numbers in this report were produced by re-running code on the uploaded data (scripts in `scripts/audit/`). Nothing is copied from the slides unless stated.

---

## 1. What exists

| Item | Status | Notes |
|---|---|---|
| `classification-regression-analysis.ipynb` | Runs top-to-bottom (given the cleaned CSV) | LR, MultinomialNB, LinearSVC, Linear/Ridge/Lasso regression |
| `ensemble_models.ipynb` | **Not reproducible as saved** | Cells were run out of order; `voting_results`, `models_for_roc`, `get_positive_scores` and the stacking `fpr/tpr` are used but never defined in the saved notebook (their cells were deleted) |
| `Clustering_and_HMM.ipynb` | Runs | 8 clustering methods + discrete/Gaussian HMM; several methodological issues (Section 6) |
| Data-cleaning script | **Missing** | All notebooks read `../datasets/processed/cleaned_train.csv`, which is not in the upload. I reconstructed it (Section 2) |
| Saved model artifacts (`.joblib/.pkl`) | **None** | No model, vectorizer or pipeline was ever saved |
| Web application | **None** | PPT 1 lists "HTML, CSS & JavaScript for the web interface", but no interface code exists |
| Tests, README, requirements | None | |

## 2. Datasets — what is actually used

| File | Rows | Labels | Used by notebooks? |
|---|---|---|---|
| `composite_train.csv` | 36,751 (6,208 exact duplicates) | 20,769 scam / 15,982 non-scam | **Yes — the only source of `cleaned_train.csv`** |
| `composite_test.csv` | 9,188 | 5,193 / 3,995 | No |
| `fraud_call.file` | 5,926 (tab-separated SMS) | 638 fraud / 5,288 normal | No |
| `English_Scam.txt` / `English_NonScam.txt` | 400 + 400 template scripts | 400 / 400 | No |

**Reconstruction of the 30,409 figure.** Lower-casing, replacing every digit run with `<number>`, deleting every character except letters/`<>`/whitespace, collapsing spaces, dropping empty rows and de-duplicating on *(cleaned_text, label)* yields exactly **30,409 rows** (19,378 scam / 11,031 non-scam; the notebook shows 19,379 / 11,030 — a 1-row difference from a regex detail in the missing script).

Findings:
- The PPTs describe "three datasets" (Scam/Non-Scam conversations, Composite, Fraud Call India). In reality **only `composite_train.csv` was used**. 5,187 of the 5,611 unique `fraud_call.file` messages are *already inside* `composite_train.csv`, so it was never a separate source.
- **Label conflicts:** de-duplicating on *(text, label)* instead of *text* kept 3 texts that appear with *both* labels (e.g. "booked ticket for pongal"). They are now removed.
- **`composite_test.csv` overlaps `composite_train.csv`:** 2,850 of its 8,743 cleaned unique rows also occur in the training file. It cannot be used as an independent test set without first removing those rows.
- **`<PHONE>` placeholder:** the PPT says phone numbers were replaced by `<PHONE>`; the cleaned text only contains `<number>`. URLs were not replaced — they were shredded into words ("www areyouunique co uk").

## 3. Reported results — reproduced

Same split (80:20, `random_state=42`, stratified), same TF-IDF (10,000 features, 1–2 grams, fitted on train only):

| Model | Reported accuracy | Reproduced accuracy | Reproduced F1 |
|---|---|---|---|
| Logistic Regression | 98.87% | 98.90% | 99.13% |
| Multinomial NB | 95.07% | 95.02% | 96.12% |
| Linear SVM | 99.42% | 99.26% | 99.42% |

The historical numbers are **reproducible to within ~0.2 percentage points**. The metric code is correct (`pos_label=1`, scam = positive class).

## 4. Leakage and evaluation checks

| Check | Result |
|---|---|
| Vectorizer fitted only on training data | ✅ Yes, in both supervised notebooks |
| Stratified split | ✅ Yes |
| Class imbalance | Moderate — 63.7% scam / 36.3% non-scam. A model that always says "scam" gets 63.7% |
| Exact duplicates across train/test | ✅ None (de-duplicated before splitting) |
| **Near-duplicates across train/test** | ⚠️ 15.2% of test conversations have a training neighbour with TF-IDF cosine ≥ 0.90 (2.8% ≥ 0.95). Removing them changes SVM accuracy only from 99.26% → 99.20%, so near-duplicates are **not** what drives the high score |
| **Test set used for model selection** | ❌ Yes. The "best model" is chosen by test-set F1 in both notebooks, and the Bagging `n_estimators` sweep (5→50) is scored on the test set. This is a (mild) form of test-set leakage. Fixed by using 5-fold CV on the training set only |
| **Dataset shortcut (the real issue)** | ❌ See Section 5 |

## 5. Why ~99% accuracy overstates real-world performance

1. **Format shortcut.** 15,987 of the 20,769 scam rows (77%) are stored as a Python-style list of utterances (`'...', '...'`). **Zero** non-scam rows have this format. A Random Forest given only raw length, word count, the count of `', '` separators, the leading quote and the `?` count — **no words at all** — reaches **92.9% accuracy**. Cleaning removes the punctuation but not the other source fingerprints.
2. **Source-specific vocabulary.** The scam dialogues are machine-translated with names like "Xiao Zhang" / "Mr. Li": `zhang` occurs in 4,857 scam rows and **0** non-scam rows; `xiao` 3,462 vs 0; `rupees` 4,152 vs 1. The word cloud in `eda.docx` shows "xiao li", "xiao zhang", "mr zhang" among the top scam terms. A model can score well by recognising the *source*, not the *scam*.
3. **Length shortcut.** A depth-3 decision tree on word count alone gets 85.6% cross-validated accuracy (majority baseline 63.7%).
4. **External test.** The 800 hand-written `English_Scam/NonScam` scripts were never used. A Linear SVM trained exactly as in the notebooks scores only **82.6% accuracy, 76.3% scam recall** on them (Logistic Regression: 82.5% / 79.8%). This is the most honest estimate available of performance on transcripts that do not come from the composite dataset.
5. **India focus is not supported by the data.** In the 30,409 rows, "otp" appears in 38 scam rows, "kyc" in 8, "upi" in 2, "aadhaar" in 0. The "Fraud Call India" claim in the slides should be softened.

**Consequence for the project:** the 99% figure is a correct *in-distribution* number and can be reported, but it must be presented together with the external-set result and the shortcut analysis. The final pipeline (Phase 4) adds the English scripts to training (half of them), keeps the other half as a held-out external test, normalises artifact tokens, and reports both numbers.

## 6. Clustering & HMM notebook issues

- **Mixed feature spaces.** K-Means is fitted on 5,000-dim TF-IDF, but GMM, all hierarchical methods, Spectral and BIRCH are fitted on a **2-D SVD projection**, and *all* silhouette/DB/CH scores are computed in that 2-D space. The K-Means scores are therefore not comparable, and 2-D silhouette scores are inflated.
- **Conclusion contradicts the table.** The markdown says "K-Means achieved the highest Adjusted Rand Index of 0.0578", but the table shows Agglomerative (Ward) at **0.1309**.
- **Single linkage** produced a 4,999 / 1 split — a degenerate result that should be called out, not ranked.
- The introduction lists **DBSCAN** and a **Scaled Forward-Backward** algorithm; neither is implemented.
- **Discrete HMM:** vocabulary is the 100 most frequent words (mostly stop-words), and every other word is silently deleted from the sequence. One HMM is trained on a mix of scam and non-scam conversations, so it cannot discriminate between them. The forward pass is unscaled (probabilities of 1e-226 on a 121-token sequence; longer sequences underflow to 0).
- **"Baum–Welch" section** re-trains the identical model with identical settings, so its output is a duplicate of the first HMM.
- **Gaussian HMM** is fitted on the 2-D SVD vectors of 30,409 *independent* conversations treated as **one 30,409-step time series**. The transition matrix it learns describes the arbitrary row order of the CSV, not conversational dynamics.

## 7. EDA document & presentation inconsistencies

| Statement | Problem | Corrected interpretation (verified) |
|---|---|---|
| "Scam conversations are generally more concise and direct" (mean 280 vs 420 words) | Contradicts the same document's box plot/KDE ("scam higher median", "scam extends toward longer conversations") | Means are pulled up by a long tail of very long non-scam transcripts (max ≈3,000 words). **Medians:** non-scam 34 words, scam 237 words. The *typical* scam transcript is longer; the class means are misleading for skewed data |
| "Correlation with label is weak, so length alone is not enough" | Pearson r (−0.13) only measures *linear* association and is distorted by outliers | Spearman ρ ≈ +0.35, and word count alone gives 85.6% accuracy. Length is strongly informative *in this dataset* — which is a dataset bias, not a property of scams |
| KDE titled "95th percentile" | x-axis extends to ~1,700 words and below 0 | Axis artefact of KDE smoothing; re-plotted with clipping |
| Word cloud includes "number number" | `<number>` placeholder rendered as a word | Placeholders now excluded from frequency plots |
| PPT: "Fraud Call India Dataset … India-focused transcripts" | `fraud_call.file` is mostly UK-style SMS; 92% already in composite | Described accurately in the new docs |
| PPT: "Linear Regression achieved low prediction errors" | Regression on a 0/1 target produces values outside [0,1]; MAE/RMSE are not meaningful for classification | Regression kept as an academic comparison only |
| PPT 1 vs PPT 2 | Different team-member names on the two decks | Please confirm which deck is yours |

## 8. Decisions carried into Phase 4

1. Rebuild the missing cleaning script and make it the single preprocessing function used for training *and* inference.
2. Add `<URL>`, `<EMAIL>`, `<PHONE>`, `<MONEY>`, `<NUMBER>` placeholders as promised in the slides; mask the speaker-name artefacts (`xiao`, `zhang`, `li`, …).
3. Model selection by 5-fold stratified CV on the training split; the test split is touched once.
4. Report three evaluations: in-distribution test, external English hold-out, and an "out-of-domain" ablation.
5. Probabilities: Logistic Regression's native probabilities and Platt-calibrated SVM (`CalibratedClassifierCV`), validated with Brier score and reliability curves. No probability is derived from raw SVM margins.
6. Advanced analysis: clustering in one consistent feature space; class-conditional HMMs with an interpretable symbol alphabet and a from-scratch scaled Forward/Backward/Viterbi verified against `hmmlearn`.

---

## Addendum — outcome of the fixes (after Phase 4)
| Decision | Outcome (from `reports/metrics.json`) |
|---|---|
| CV-based selection | Calibrated Linear SVM selected: CV F1 99.44 ± 0.07 (stacking 99.43, LR 99.36) |
| In-distribution test | Default model accuracy 99.28%, F1 99.43%, 30 false negatives / 14 false positives out of 6,079 |
| English scripts half in training | External held-out accuracy 98.5% (was 82.2% when none were in training) |
| Masking source names | No measurable effect on test scores (LR external accuracy 98.49% either way). Kept because it removes an obvious shortcut at no cost |
| Near-duplicates (final cleaning) | 13.7% of test rows; removing them: 99.28% → 99.18% |
| Calibration | Brier 0.0058 (SVM), 0.0068 (LR), 0.0335 (NB), 0.1415 (AdaBoost) |
| Clustering in one 50-D space | Silhouettes ≈ 0.03–0.08 (the original 0.58 was inflated by 2-D scoring); best ARI 0.077 |
| Class-conditional HMMs | 86.2% in-distribution vs 83.8% order-blind baseline; 74.1% vs 75.8% external |
| Known failure kept visible | Genuine bank fraud-alert call scored >99% scam; documented as an `xfail` test |

Note: §3–§5 of this audit used the *original* cleaning. Numbers elsewhere use the final pipeline, so small differences are expected (e.g. medians 237/34 here vs 231/33 in the app, out-of-domain SVM 82.6% vs 82.2%).
