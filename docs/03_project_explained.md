# The Project, Explained End to End

Each topic starts with **In plain words** (what you'd say to a non-specialist) and then **Technically** (what you'd say to faculty).

---

## 1. Problem statement
**In plain words.** Scammers phone people pretending to be the bank, police or a courier, rush them and ask for an OTP or a payment. Victims often realise it only later. We built a tool where you paste the conversation and it tells you how risky it was, why, and what to do.

**Technically.** A binary text-classification task at the *conversation* level (scam = 1, non-scam = 0), extended with (a) a calibrated probability and three-level risk mapping, (b) per-prediction explanations, and (c) a transparent rule layer for known social-engineering tactics.

## 2. Why it is needed
Phone scams have no e-mail headers or URLs to inspect; the evidence is in *what is said*. Manual review of long transcripts is slow and inconsistent. Literature (see `02_literature_survey.md`) shows strong in-domain models but little attention to calibration, explanation and dataset shift — the parts a real user depends on.

## 3. Dataset
**In plain words.** About 30,000 labelled conversations and messages, roughly two-thirds scams, plus 800 hand-written English example scripts we used to check the model on different-looking text.

**Technically.**
- `composite_train.csv`: 36,751 rows → 30,392 unique after cleaning (63.7% scam). It mixes translated scam dialogues, long call transcripts and SMS.
- English scripts: 794 unique → 397 added to training, 397 kept as an external test.
- `composite_test.csv` (unseen rows only) and `fraud_call.file` (unseen rows only) as extra tests.
- **Audit finding:** the sources differ systematically between classes (format, length, translated names). A no-words model gets 92.8%. We report this openly and test on external data.

## 4. Data preprocessing
**In plain words.** We tidy the text the same way every time: lower-case, swap phone numbers/links/amounts for tags like `<PHONE>`, remove punctuation and names that only reveal where the data came from, and delete empty or duplicate rows.

**Technically.** `src/scam_analyzer/preprocessing.py::clean_text` — regex replacement of e-mails, URLs, currency amounts, 10–13-digit phone numbers, payment-app names and remaining numbers with `__x__` tokens (kept by the TF-IDF tokenizer); apostrophes joined; non-letters removed; source-artifact tokens dropped. `data.py` drops empties, conflicting-label texts and duplicates, then splits (80:20 stratified, `random_state=42`). The identical function runs at inference — no train/serve skew.

## 5. Exploratory data analysis
**In plain words.** We looked at how long conversations are, which words and phrases each class uses, and how features relate to the label — and corrected some earlier conclusions.

**Technically.**
- Medians matter for skewed data: scam median 231 words vs non-scam 33; means (282 vs 426) point the other way because of a long tail of very long non-scam transcripts.
- Pearson r(length, label) ≈ −0.13 looks weak, but Spearman ρ ≈ +0.35, and word count alone gives ≈86% accuracy → strong, non-linear association.
- Scam bigrams: "customer service", "id number", "phone number", "provide id", "safety funds"; non-scam: conversational topics. The distinctive-term list exposes source topics (e.g. "coronavirus" only in non-scam).

## 6. TF-IDF
**In plain words.** We turn each conversation into a list of numbers: one per word or two-word phrase, high when the phrase is frequent in this conversation but rare overall.

**Technically.** `TfidfVectorizer(max_features=10000, ngram_range=(1,2))`. tf = raw count, idf = ln((1+N)/(1+df)) + 1, rows L2-normalised. Fitted on training data only (inside each CV fold during model selection). A CV study confirmed the original configuration.

## 7. Classification algorithms
| Model | One-line intuition | Key formula / idea |
|---|---|---|
| Logistic Regression | Weighted vote of words, squashed to a probability | P = σ(w·x + b), minimise log-loss + L2 |
| Multinomial NB | How likely are these words under "scam" vs "non-scam"? | log P(c\|x) ∝ log P(c) + Σ xᵢ log P(wᵢ\|c) |
| Linear SVM | Widest possible gap between classes | minimise ½‖w‖² + C Σ hinge(yᵢ(w·xᵢ+b)) |

## 8. Ensemble methods
- **Hard voting**: majority of LR, NB, SVM labels.
- **Soft voting**: average of LR, NB and *calibrated* SVM probabilities.
- **Stacking**: LR meta-model trained on 5-fold out-of-fold predictions of LR, NB, SVM.
- **Bagging**: 50 decision trees on 80% bootstrap samples, averaged (variance reduction).
- **AdaBoost**: 50 decision stumps trained sequentially on re-weighted errors (bias reduction).
On sparse text, the linear models and their combinations are best; tree ensembles trail.

## 9. Clustering (supplementary)
**In plain words.** Without using labels, we let the computer group similar conversations. It finds topics, not "scam vs not".

**Technically.** TF-IDF → TruncatedSVD(50) → L2-normalise (LSA). K-Means, GMM, Agglomerative (Ward/single/complete/average), Spectral and BIRCH, all with k = 2 and all scored in the same space (silhouette, Davies–Bouldin, Calinski–Harabasz) and against labels (ARI, NMI, purity). K-Means with k = 8 gives interpretable topic clusters.

## 10. Hidden Markov Model (supplementary)
**In plain words.** Scams tend to follow a script: introduce authority → describe a problem → pressure → ask for something. An HMM models such hidden "stages" from the sequence of what's said.

**Technically.** Each 12-word window → one of 11 symbols (NEUTRAL, AUTHORITY, ACCOUNT, URGENCY, REWARD, PERSONAL_INFO, CREDENTIAL, PAYMENT, LINK, SECRECY, REMOTE). Two 3-state categorical HMMs (scam / non-scam) trained with Baum–Welch; classification by log-likelihood ratio; compared with an order-blind NB on symbol counts. Forward, Backward, posterior and Viterbi are implemented from scratch with scaling and verified against `hmmlearn`.

## 11. Model evaluation
- 5-fold stratified CV on the training split → model selection (F1 of scam class).
- Test sets used once: in-distribution (composite 20%), external (English scripts 50%), composite_test unseen rows, fraud_call unseen rows.
- Metrics: accuracy, precision, recall, F1, specificity, confusion matrix, ROC-AUC, average precision; for probabilistic models Brier score, log-loss, ECE and reliability curves.
- Ablations: historical reproduction, out-of-domain, name masking, format/length shortcut, near-duplicate filtering.

## 12. Risk assessment
**In plain words.** The model gives a percentage. Under 35% is Low, 35–70% Medium, 70%+ High. If the model says Low but someone clearly asked for an OTP, we bump it to Medium and say why.

**Technically.** Calibrated P(scam) from LR (native) or SVM (Platt scaling, `CalibratedClassifierCV(cv=5, method="sigmoid")`). Thresholds and the escalation rule are in `config/settings.toml`. The decision label uses 0.5; the MEDIUM band starts lower to favour recall.

## 13. Explanation
- **Model evidence (exact):** contribution of term i = TF-IDF value × weight. Positive → toward scam, negative → toward legitimate; they sum exactly to the model's score.
- **Rule indicators (transparent):** 10 categories of phrase patterns with severity, why it matters and advice; "safe signals" suppress overlapping credential flags.
- **Highlighted transcript:** red = rule flag, cyan = scam-leaning term, green = legitimate-leaning term.

## 14. Application architecture
Streamlit multi-page app (`app/streamlit_app.py`) over the `scam_analyzer` package. Analyzer and models are cached; reports are precomputed JSON. Six pages: Home, Scam detection, Model comparison, Dataset explorer, Advanced ML, About. No external API; inference ≈ milliseconds.

## 15. Limitations
Dataset shortcuts; bag-of-words without intent, speaker roles or negation; little Indian-context training data; text-only; evadable by rephrasing; probabilities calibrated to a 64%-scam mix.

## 16. Future scope
India-focused conversational data (ICFD-31k, Hinglish), speaker-aware and streaming detection, calibrated multilingual transformer baseline, prevalence-aware recalibration, feedback loop for active learning.
