# Quick Revision Sheet (read the night before)

## Project in three sentences
Post-call scam risk analyzer: transcript → cleaned text → TF-IDF (10k uni+bigrams) → calibrated linear model → P(scam) → Low/Medium/High.
Each result comes with exact word contributions (model evidence) and, separately, rule-based red flags plus advice.
We audited the original work, reproduced its 99% numbers, found dataset shortcuts, and report external / out-of-domain results honestly.

## Numbers to remember
| Item | Value |
|---|---|
| Raw composite rows | 36,751 |
| Original cleaned count (reproduced) | 30,409 |
| New cleaned unique conversations | 30,392 (19,373 scam / 11,019 non-scam, 63.7% scam) |
| Training split | 24,710 (composite 24,313 + English 397) |
| Test sets | composite 6,079 · English 397 · composite_test unseen 5,887 · fraud_call unseen 422 |
| Notebook accuracy (reported → reproduced) | LR 98.87 → 98.90 · NB 95.07 → 95.02 · SVM 99.42 → 99.26 |
| Format/length-only model accuracy | 92.8% (majority baseline 63.7%) |
| Out-of-domain (trained without English scripts) | SVM 82.2% acc / 78.7% recall; LR 82.5% / 80.2% |
| Near-duplicates in test (cos ≥ 0.9) | 13.7%; removing them: 99.28% → 99.18% |
| Risk thresholds | LOW < 0.35 ≤ MEDIUM < 0.70 ≤ HIGH; decision 0.50 |
| Default model / selection | see `reports/metrics.json` → `best_model`; chosen by 5-fold CV F1 on training data |

## Formulas
- TF-IDF: tf(t,d) × [ln((1+N)/(1+df(t))) + 1], then L2-normalise the row.
- Logistic: P = 1 / (1 + e^−(w·x+b)); loss = −[y log p + (1−y) log(1−p)] + ‖w‖²/(2C).
- Naive Bayes: log P(c|x) ∝ log P(c) + Σ xᵢ log θ_{c,i}, θ smoothed with α.
- SVM: min ½‖w‖² + C Σ max(0, 1 − yᵢ(w·xᵢ+b)).
- Platt scaling: P = 1 / (1 + e^(A·f(x)+B)), A, B fitted on held-out folds.
- Precision TP/(TP+FP) · Recall TP/(TP+FN) · F1 = 2PR/(P+R) · Specificity TN/(TN+FP).
- Brier = mean (p − y)²; ECE = Σ_bins (n_b/N)|mean p_b − mean y_b|.
- Silhouette s = (b − a)/max(a, b); ARI adjusts Rand index for chance (0 = random, 1 = identical).
- HMM: Forward α_t(j) = [Σ_i α_{t−1}(i)a_ij] b_j(o_t); Viterbi δ_t(j) = max_i[δ_{t−1}(i) + log a_ij] + log b_j(o_t); scaling: log P = Σ log c_t.

## Design decisions (one line each)
- **Model selection by CV, not test** — avoid test-set leakage (original notebook didn't).
- **Calibrated SVM** — SVM margins are not probabilities.
- **Soft voting now includes SVM** — possible after calibration.
- **Placeholders** for URL/phone/money/number — keep the type, drop the value.
- **Mask source names** (zhang, xiao…) — they identify the data source, not scams (no measurable change in test scores, but removes an obvious shortcut).
- **English scripts split 50/50** — some domain coverage in training, an external test kept aside.
- **MEDIUM from 0.35** — false negatives are costlier than false positives.
- **Rules separate from model** — transparent, cover India cues, never presented as proof.
- **HMM on stage symbols, class-conditional** — interpretable, discriminative, no dropped words.

## Corrected claims (don't repeat the old ones)
- ❌ "Scam conversations are more concise" → ✅ lower *mean* (282 vs 426) but much higher *median* (231 vs 33 words).
- ❌ "Length is weakly related to the label" → ✅ weak *Pearson*, moderate *Spearman* (+0.35); length alone ≈ 86% accurate (a dataset bias).
- ❌ "Three datasets incl. Fraud Call India" → ✅ only composite_train was used; fraud_call is ~92% contained in it.
- ❌ "<PHONE> placeholder used" → ✅ now true (it wasn't in the original cleaning).
- ❌ "K-Means best ARI" → ✅ Ward had the highest ARI in the original table.
- ❌ "Linear regression achieved low error" → ✅ regression on 0/1 labels is not a valid risk score.

## Literature in one breath
Derakhshan '21 (scam signatures, ACM) · Tsinganos '22 (persuasion CNN, IEEE Access) · Moussavou Boussougou & Park '23 (Korean vishing CNN-BiLSTM, Mathematics) · Jáñez-Martino '23 (dataset shift, AI Review) · Ghourabi & Alohaly '23 (GPT-3 emb + voting, Sensors) · Wood '23 (scam stages + HMM, arXiv) · Ai '24 (ConvoSentinel, EMNLP) · Salman '24 (SMS evasion, IEEE Access) · Dahiphale '24 (UPI LLM, arXiv) · Ma '25 (TeleAntiFraud-28k, ACM MM) · Ahuja '26 (ICFD-31k India, IJCAI) · Lochstampfor & Roy '26 (TF-IDF beats transformers, arXiv) · Uddin '26 (explainable RoBERTa, Computer Networks).

**Gap:** lightweight + calibrated + exact explanations separated from rules + honest out-of-distribution evaluation, in one local tool. Not "first scam detector", not "India-trained".

## Commands
```bash
pip install -r requirements.txt
python scripts/train_models.py        # trains + evaluates 8 models → models/, reports/metrics.json
python scripts/run_analysis.py        # EDA + clustering + HMM → reports/
streamlit run app/streamlit_app.py    # the app
pytest                                # tests
python scripts/demo_cli.py "text"     # terminal fallback
```
