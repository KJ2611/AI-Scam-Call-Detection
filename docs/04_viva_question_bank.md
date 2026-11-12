# Faculty Cross-Question Bank (60 questions)

Numbers quoted are from `reports/metrics.json` produced by `scripts/train_models.py` (re-run gives the same values; `random_state = 42`). If you retrain with different data, re-check them in the app's Model Comparison page.

Legend: **[B]** basic · **[I]** intermediate · **[A]** advanced / challenging

---

## A. Problem, motivation, scope

**1. [B] What problem does your project solve?**
People often realise only *after* a call that it was a scam. We take the transcript of a finished call or chat and give (1) a scam / non-scam prediction, (2) a calibrated probability and Low/Medium/High risk level, (3) the words and manipulation tactics that drove the result, and (4) what to do next.

**2. [B] Why "post-call" and not real-time?**
Post-call needs only the full transcript, so a lightweight text model is enough and we can evaluate cleanly on whole conversations. Real-time detection needs streaming ASR, partial-transcript models and latency guarantees; we list it as future work (ICFD-31k, IJCAI 2026, studies exactly that).

**3. [I] Isn't this just spam detection?**
Related, but different units and cues. Spam filters classify one short message; we classify a whole conversation where the scam builds up over turns (authority → problem → urgency → request). That is why we also show stage symbols and an HMM over the conversation. The underlying classifier (TF-IDF + linear model) is the same family used in spam filtering, and we say so.

**4. [I] Who would use it?**
A person reviewing a suspicious call, a bank's fraud-awareness team reviewing reported calls, or a teacher/trainer using it to show manipulation tactics. It is decision support, not an automatic blocker.

**5. [A] Can your system say a call *is* fraud?**
No. It outputs a model-assisted risk assessment. The model reacts to word patterns; it does not verify identities, numbers or transactions. The app says this on every result.

## B. Dataset

**6. [B] What data did you use?**
Primary: `composite_train.csv` — 36,751 rows → 30,392 unique conversations after cleaning (19,373 scam, 11,019 non-scam, 63.7% scam). Plus 800 English scam/non-scam scripts (794 after de-duplication), split 50/50 into extra training data and an external test set. `composite_test.csv` and `fraud_call.file` are used only as extra test sets, after removing rows that also appear in training.

**7. [I] Your slides say 30,409 conversations; the app says 30,392. Why?**
30,409 was produced by the original cleaning (digits → `<number>`, de-duplicate on text+label). We reproduced that number exactly. Our new cleaning adds placeholders (URL/phone/money), masks source names and also removes texts that carried *both* labels, so slightly more rows collapse into duplicates.

**8. [I] Is the dataset balanced? Does it matter?**
Moderately imbalanced (≈64/36). We used stratified splits and stratified CV, report precision/recall/F1 (not just accuracy), and the majority-class baseline (63.7%) is shown so accuracy can be judged against it.

**9. [A] You found the dataset has "shortcuts". Explain.**
77% of scam rows are stored as a Python-style list of utterances; no non-scam row is. Scam dialogues contain translated names ("zhang" in 4,856 scam rows, 0 non-scam). Many non-scam rows are short SMS. A Random Forest given only length/format features — no words — reaches 92.8% accuracy. So high in-distribution accuracy partly measures *which source* a row came from.

**10. [A] Then is your 99% meaningless?**
It is a correct in-distribution number but optimistic for real calls. That's why we report: (a) an external test on hand-written English scripts, (b) an out-of-domain test where those scripts were never seen (82.2% accuracy, 78.7% recall for the calibrated SVM), (c) masking of source names (which, honestly, changed test scores by < 0.05 points — the model was not relying on names alone), and (d) the shortcut experiment. Jáñez-Martino et al. (2023) and Ahuja et al. (2026) report the same gap between in-domain and shifted data.

**11. [I] Did you have duplicates or leakage between train and test?**
Exact duplicates are removed *before* splitting. 13.7% of test conversations have a near-duplicate (cosine ≥ 0.90) in training; removing them changes the default model's accuracy only from 99.28% to 99.18%, so they don't explain the score. `composite_test.csv` shared ≈2,850 rows with the training file — we remove them before using it as a test.

**12. [I] How did you protect sensitive information?**
Phone numbers, e-mails and links are replaced by placeholders in model input and masked in displayed samples. The app runs locally; no transcript is sent anywhere.

## C. Preprocessing & features

**13. [B] What preprocessing steps do you apply?**
Lower-case; replace e-mails, URLs, money amounts, phone numbers, payment-app names and other numbers with placeholder tokens; remove punctuation; mask source-identifying names; collapse whitespace; drop empty, duplicate and conflicting-label rows. The *same* function is used at training and prediction time.

**14. [I] Why replace numbers with placeholders instead of deleting them?**
"Call 98765 43210" and "pay Rs 5,000" are informative *as a type* (a phone number, an amount), not as specific values. Placeholders keep the signal while preventing the model from memorising particular numbers.

**15. [I] Why didn't you remove stop-words?**
Bigrams such as "your account", "need to", "do not" contain stop-words and carry meaning. Stop-words are removed only for EDA word-frequency charts.

**16. [B] What is TF-IDF?**
Term frequency × inverse document frequency. A word scores high in a document if it is frequent there but rare across the collection. idf(t) = ln((1+N)/(1+df(t))) + 1 in scikit-learn; rows are L2-normalised.

**17. [I] Why 10,000 features and (1,2)-grams?**
Kept from the original notebooks after a 3-fold CV study on the training data: 20,000 features, sublinear tf or min_df=2 changed F1 by ≤0.05 points; unigrams alone were ≈0.2 points worse. So the simpler original setting stays.

**18. [A] Why TF-IDF instead of BERT?**
(1) Runs on a laptop in milliseconds with no GPU or API; (2) linear models on TF-IDF give *exact* per-word explanations; (3) with ~25k training conversations, TF-IDF models are competitive — Lochstampfor & Roy (2026) found TF-IDF + XGBoost beat fine-tuned DistilBERT/Longformer on conversational smishing. The limitation — no word order or intent understanding — is stated openly; a calibrated transformer is future work.

**19. [A] Was the vectorizer fitted on test data?**
No. In CV, TF-IDF is inside a scikit-learn `Pipeline`, so it is re-fitted on each training fold. For the final model it is fitted on the training split only, then used to transform test sets.

## D. Models

**20. [B] Which models did you compare and why each?**
LR (linear, probabilistic, explainable), Multinomial NB (classic fast text baseline), Linear SVM (strongest linear text classifier), Hard voting, Soft voting, Stacking (combine the three), Bagging of decision trees and AdaBoost (tree ensembles for contrast). All were in the original notebooks; none were added for show.

**21. [B] How does Logistic Regression work?**
Computes z = w·x + b and P(scam) = 1/(1+e^(−z)); weights are learned by minimising log-loss (with L2 regularisation, C = 10 chosen by CV).

**22. [B] How does Multinomial Naive Bayes work?**
Applies Bayes' rule assuming words are independent given the class: log P(scam|x) ∝ log P(scam) + Σ xᵢ log P(wordᵢ|scam). Fast and strong on small data, but the independence assumption makes its probabilities over-confident (visible in our reliability plot).

**23. [B] How does a linear SVM work?**
Finds the hyperplane w·x + b = 0 that separates the classes with the largest margin, using hinge loss; C trades margin width against training errors.

**24. [I] What is the difference between hard and soft voting?**
Hard voting takes the majority of predicted labels. Soft voting averages predicted probabilities, so a very confident model counts more. Soft voting needs probabilities from every member — the original notebook had to drop the SVM; we include it after calibration.

**25. [I] How does stacking differ from voting?**
Stacking trains a meta-model (here LR) on the base models' *out-of-fold* predictions (cv = 5), so it learns how much to trust each model instead of weighting them equally.

**26. [I] Bagging vs boosting?**
Bagging trains many models in parallel on bootstrap samples and averages them → reduces variance. Boosting (AdaBoost) trains models sequentially, up-weighting misclassified examples → reduces bias. Our AdaBoost uses depth-1 stumps, so it's weaker on 10,000 sparse features.

**27. [A] Why are the tree ensembles worse here?**
High-dimensional sparse text suits linear models: each tree split looks at one word at a time, and 50 trees cannot cover 10,000 features as efficiently as one weight per feature. That's the usual finding for bag-of-words text.

**28. [I] Which model is best and how did you decide?**
We choose by mean 5-fold CV F1 on the training split among probabilistic models — test sets are not used for the choice. The calibrated Linear SVM had the highest CV F1 (99.44 ± 0.07%), with stacking (99.43%) and LR (99.36%) close behind. Those differences are within CV noise (std ≈0.1), so we also value simplicity and explainability, which the linear SVM has.

**29. [A] The original notebook picked the best model by test-set F1. What's wrong with that?**
The test set then influences a decision, so its score is no longer an unbiased estimate (mild selection bias / "test-set leakage"). The bagging `n_estimators` sweep was also scored on the test set. We fixed both by selecting with cross-validation on training data only.

## E. Evaluation

**30. [B] Define precision, recall, F1.**
Precision = TP/(TP+FP): of conversations flagged scam, how many really were. Recall = TP/(TP+FN): of real scams, how many we caught. F1 = harmonic mean of the two.

**31. [B] Why is accuracy not enough?**
With 64% scam, a model that always says "scam" gets 64% accuracy and catches all scams but is useless. Precision/recall/F1 and the confusion matrix show what kind of errors are made.

**32. [I] Which error is worse here, FP or FN?**
False negatives — a scam called safe — because the user relaxes exactly when they should be careful. A false positive costs a little extra caution. Hence the MEDIUM band starts at 0.35 (below 0.5), and recall is shown next to F1.

**33. [I] What is ROC-AUC? Average precision?**
ROC-AUC = probability a random scam gets a higher score than a random non-scam (threshold-free ranking quality). Average precision summarises the precision–recall curve; it's more informative under class imbalance.

**34. [A] What is cross-validation and why stratified?**
Split training data into k folds; train on k−1, validate on the remaining one, rotate, average. Stratified keeps the 64/36 ratio in every fold so scores are comparable. It gives a mean ± std, which tells us whether two models really differ.

**35. [A] What are the regression experiments and why not use them for risk?**
Linear/Ridge/Lasso were fitted to the 0/1 label. Their outputs fall outside [0,1] (the notebook shows 1.22 and −0.16), squared error is the wrong loss for classification, and they are not calibrated. Logistic regression is the correct linear model for a probability.

## F. Risk score, calibration & explanations

**36. [B] How is the risk level computed?**
From the calibrated probability: LOW < 0.35 ≤ MEDIUM < 0.70 ≤ HIGH. Thresholds are in `config/settings.toml` and can be changed without retraining.

**37. [I] Why can't you use the SVM's decision_function as a confidence?**
It is a signed distance to the hyperplane in arbitrary units, not a probability; 2.0 does not mean "80% sure". We use Platt scaling (`CalibratedClassifierCV`, sigmoid, 5-fold) to map it to probabilities learned on held-out folds.

**38. [A] How do you know the probabilities are calibrated?**
On the test set we report the Brier score, expected calibration error (ECE) and a reliability curve (predicted vs observed scam rate per bin). Calibrated SVM (Brier 0.0058) and LR (0.0068) lie close to the diagonal; Naive Bayes (0.0335) and especially AdaBoost (0.1415, scores bunched near 0.5) are poorly calibrated.

**39. [A] Calibrated to what prevalence?**
To the training mix (~64% scam). In real life most calls are not scams, so the same score would correspond to a lower true probability. This is listed as a limitation; recalibration with real prevalence is future work.

**40. [I] How do you explain a prediction?**
For linear models, score = b + Σ xᵢwᵢ. Each term's xᵢwᵢ is its exact contribution. We plot the largest positive (toward scam) and negative (toward legitimate) terms. For ensembles we show LR's evidence and say so.

**41. [I] What is the rule layer and why separate it?**
Hand-written phrase patterns grouped into 10 tactics (OTP/PIN request, remote access, payment request, personal info, account verification, urgency/threat, impersonation, reward lure, secrecy, links). They are transparent and cover India-specific cues the data lacks, but they are keyword matches, so we show them separately from model evidence and never call them proof.

**42. [A] Can the rule layer change the prediction?**
It never changes the probability or the SCAM/NON-SCAM label. If the model says LOW but a HIGH-severity rule matches (e.g. "share the OTP"), the displayed risk is raised to MEDIUM and the reason is printed. This is configurable.

**43. [A] What about negation — "we will never ask for your OTP"?**
TF-IDF can't understand negation. The rule layer has "safe-signal" patterns that detect such phrases, suppress the overlapping credential flag and show them as reassuring. It's a targeted patch, not general negation handling.

**44. [A] Does your model understand intent?**
No. It learns which words and word pairs co-occur with scam labels. We avoid claiming understanding.

## G. Clustering & HMM

**45. [B] Why include clustering if you have labels?**
To explore structure without labels: K-Means with k = 8 finds topics (bank verification, prizes, casual chat…) and shows which topics the scam data is built from. It is supplementary.

**46. [I] Can a cluster be called "scam"?**
No — cluster IDs are arbitrary. We measure agreement with labels using ARI, NMI and purity; agreement is low, as expected, because clustering groups by topic/style.

**47. [I] What did you fix in the original clustering?**
All methods now run and are scored in the same 50-D LSA space (the original scored K-Means in a different space from the others), the degenerate single-linkage split is flagged, and the conclusion text is generated from the table (the original named the wrong method as best ARI).

**48. [I] Silhouette vs Davies–Bouldin vs Calinski–Harabasz?**
Silhouette (−1..1, higher better) compares distance to own vs nearest other cluster. Davies–Bouldin (lower better) is the average worst-case ratio of within-cluster spread to between-centre distance. Calinski–Harabasz (higher better) is between-cluster dispersion over within-cluster dispersion.

**49. [B] What is a Hidden Markov Model?**
A model with hidden states that follow a Markov chain (transition matrix A, start π) and emit observations (emission matrix B). We see only the observations.

**50. [I] What are your hidden states and observations?**
Observations: one symbol per 12-word window — the manipulation category present (AUTHORITY, ACCOUNT, URGENCY, CREDENTIAL, PAYMENT, …) or NEUTRAL. Hidden states (3): learned "phases" such as authority/account set-up vs pressure/request, named after their dominant emissions.

**51. [I] Explain Forward, Backward and Viterbi.**
Forward: P(observations | model), summing over all state paths with dynamic programming. Backward: probability of the remaining observations from each state; combined with Forward gives the posterior P(state at t | all observations). Viterbi: the single most likely state path (max instead of sum, with back-pointers). Baum–Welch: EM that learns A, B, π using Forward–Backward.

**52. [A] Why scaling in the forward algorithm?**
Multiplying many probabilities underflows to 0 — the original notebook already reached 1e-226 at 121 steps. We normalise at each step and sum log-scales; our test shows the unscaled value becomes exactly 0 at T = 1,200 while the scaled log-likelihood is finite.

**53. [A] How do you use HMMs for classification?**
Train one HMM on scam sequences and one on non-scam sequences; classify a new sequence by the log-likelihood ratio (+ log prior). We compare with Naive Bayes on symbol *counts* to see whether *order* adds anything.

**54. [A] Why didn't you keep the Gaussian HMM?**
The original fitted it on 30,409 *independent* conversations as if they were one time series, so the transitions describe CSV row order. A Gaussian HMM needs a real continuous time series within each conversation.

**55. [A] How do you know your HMM code is correct?**
Unit tests compare our Forward log-likelihood, Viterbi path and posteriors with `hmmlearn` on random models (differences < 1e-8), and the analysis script checks 25 real sequences.

## H. Literature & research gap

**56. [I] Summarise the literature in one minute.**
Phone-scam detection started with rule/semantic signatures (Derakhshan 2021) and persuasion classifiers (Tsinganos 2022), moved to deep models on vishing transcripts (Moussavou Boussougou & Park 2023) and now to LLM/multimodal datasets (TeleAntiFraud-28k, ConvoSentinel, ICFD-31k). Reported in-domain scores are 98–99%+, but dataset-shift studies (Jáñez-Martino 2023; Salman 2024) and cross-domain tests (ICFD-31k: F1 99.4 → 93.0) show these don't transfer. Explanations are mostly post-hoc or LLM-generated, and calibration is rarely reported.

**57. [A] What exactly is your research gap? Are you the first?**
We are not the first scam-conversation classifier. The gap we address is the *combination*: lightweight local classification + calibrated probability with documented thresholds + exact model evidence kept separate from transparent rules + honest out-of-distribution evaluation and shortcut audit. We scope this claim to the papers we reviewed.

**58. [A] Your title says India-focused. Is it?**
Only partly, and we say so. The training data has almost no India-specific vocabulary (OTP in 45 scam rows, Aadhaar in 0). India-specific cues (OTP, UPI, KYC, Aadhaar, AnyDesk, "digital arrest", 1930 helpline advice) are in the rule layer and recommendations. ICFD-31k (IJCAI 2026, English + Hinglish) is the natural next training set.

## I. Architecture, limitations, future work

**59. [B] Describe the application architecture.**
`data.py` cleans/splits → `train.py` cross-validates, fits and evaluates 8 models, saving `models/*.joblib` and `reports/metrics.json` → `risk.ScamAnalyzer` loads them and runs clean → TF-IDF → model → calibrated probability → risk + `explain.py` evidence + `indicators.py` rules → Streamlit pages. EDA and clustering/HMM reports are precomputed JSON. Everything runs offline.

**60a. [A] Show me a case where your model is wrong.**
The 'Genuine bank fraud-alert call' example: a real bank asks "was this payment yours?" and says "we will never ask for your OTP". The SVM gives >99% scam, because in the training data almost every conversation mentioning a bank card is a scam (dataset bias + no negation). The rule layer gets it right: no credential *request*, three safe signals. We kept this as a documented failure (an `xfail` test) instead of patching it, because a rule that *lowers* risk when it sees "we never ask for OTP" would be trivially exploitable by scammers.

**60. [A] Give three honest limitations and how you'd fix them.**
(1) Dataset shortcuts → train/evaluate on cleaner, source-matched data (e.g. ICFD-31k, consented real transcripts) and keep an external test. (2) Bag-of-words can't model intent, speaker roles or negation → speaker-aware features or a calibrated transformer, evaluated with the same protocol. (3) Calibrated to a 64% scam training mix → recalibrate for realistic prevalence and validate on field data.
