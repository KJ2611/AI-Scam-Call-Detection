# Literature Survey — Scam Conversation Detection (2021–2026)

Every paper below was checked against its publisher page, ACL Anthology, IJCAI proceedings or arXiv (November 2026). **Peer-reviewed** and **preprint (not peer-reviewed)** status is marked explicitly. Where a number is quoted, it is the number the paper itself reports; where the abstract or proceedings page did not give a number, none is invented here.

> **How to use this in the viva:** read Section 1 (one-paragraph summaries you can say aloud), then the comparison table (Section 2), then the research gap (Section 3). Section 5 has the full references.

---

## 1. Paper-by-paper analysis

### P1. Derakhshan, Harris & Behzadi (2021) — Scam signatures for telephone social engineering
- **Venue:** ACM Workshop on Security and Privacy Analytics (IWSPA @ CODASPY) 2021, pp. 67–73. **Peer-reviewed.** DOI: 10.1145/3445970.3451152
- **Problem:** Phone scams are less studied than phishing e-mail and have no metadata (headers, URLs) to inspect — detection must use the *content* of the conversation.
- **Dataset:** A new telephone-scam dataset written by volunteers who re-wrote real scam examples from official awareness websites in their own words; the authors describe it as the first such dataset.
- **Method / algorithms:** ASsET (Anti-Social Engineering Tool). A scam type is characterised by a "signature" — a set of *speech acts* (e.g., "request SSN", "threaten arrest"). Word embeddings test whether the semantic content of each speech act appears in the conversation.
- **Metrics / findings:** Reports that the method distinguishes scam from non-scam calls with high accuracy.
- **Limitations:** Small, volunteer-written (not real) calls; signatures must be hand-designed per scam type.
- **Relevance to us:** Direct precedent for our **rule-based indicator layer** (OTP request, authority impersonation, urgency…) — our indicator categories are essentially coarse speech-act signatures. It also shows that conversation-level phone-scam detection is an established problem, so we do not claim novelty for the task itself.
- **Say it simply:** "They described each scam as a checklist of things the scammer *does* — ask for your SSN, threaten you — and checked whether a call contains those actions."

### P2. Tsinganos, Mavridis & Gritzalis (2022) — Recognising persuasion in chat-based social engineering
- **Venue:** IEEE Access, vol. 10, pp. 108517–108529, 2022. **Peer-reviewed.** DOI: 10.1109/ACCESS.2022.3213681
- **Problem:** Chat-based social engineering (CSE) relies on persuasion; early recognition of persuasive sentences could warn the victim.
- **Dataset:** CSE Corpus, annotated for Cialdini's persuasion principles.
- **Method / algorithms:** A convolutional neural network over word embeddings (CSE-PUC) classifies whether a sentence carries a persuasive payload, producing a probability distribution.
- **Findings:** Reports that the classifier can recognise persuasion methods in chat dialogues.
- **Limitations:** Sentence-level, enterprise-chat setting; a single "persuasive / not" decision, not a conversation-level scam verdict.
- **Relevance:** Supports our choice to surface *manipulation tactics* (urgency, authority, reward) as explanations, not just a label. The same group later built a full late-fusion system (CSE-ARS, IEEE Access 2024).

### P3. Moussavou Boussougou & Park (2023) — Attention-based 1D CNN-BiLSTM for Korean voice phishing
- **Venue:** *Mathematics* (MDPI) 11(14):3217, 2023. **Peer-reviewed.** DOI: 10.3390/math11143217
- **Problem:** Detect voice-phishing (vishing) from call transcripts in Korean, where few studies and small datasets exist.
- **Dataset:** KorCCVi v2 — labelled real-world Korean call-content vishing transcripts.
- **Method / algorithms:** FastText embeddings → 1D CNN → BiLSTM → hierarchical attention; compared against ML and DL baselines. Framed as combining *data-centric* and *model-centric* AI.
- **Findings:** Reports that the hybrid architecture extracts contextually rich features and outperforms the compared baselines on KorCCVi v2.
- **Limitations:** Single language; deep model with limited interpretability; small dataset.
- **Relevance:** Closest published analogue of our task (transcript-level vishing classification). Several long non-dialogue scam transcripts in our composite data read like translated Korean financial-fraud calls, so this line of work is the natural comparison for our classical TF-IDF baseline.

### P4. Jáñez-Martino et al. (2023) — Spam detection and the dataset-shift problem
- **Venue:** *Artificial Intelligence Review* 56(2):1145–1173, 2023. **Peer-reviewed (review article).** DOI: 10.1007/s10462-022-10195-4
- **Problem:** Published spam filters report very high accuracy, yet users still receive fraud — why?
- **Method:** Review of spammer strategies and ML filters, plus an empirical study of what happens when *dataset shift* is ignored (covariate, prior-probability and concept shift).
- **Findings:** Ignoring dataset shift can severely degrade real generalisation; they report error rates reaching 48.81%.
- **Limitations:** E-mail domain; review rather than a new detector.
- **Relevance:** **The most important paper for defending our evaluation.** Our audit found the same phenomenon: 99% in-distribution accuracy, ~83% on the unseen English scripts. This paper is the citation for "high benchmark accuracy ≠ real-world accuracy".

### P5. Ghourabi & Alohaly (2023) — Transformer embeddings + ensemble learning for SMS spam
- **Venue:** *Sensors* 23(8):3861, 2023. **Peer-reviewed.** DOI: 10.3390/s23083861
- **Dataset:** SMS Spam Collection (the classic UCI dataset — which is also embedded in our composite data).
- **Method / algorithms:** GPT-3 text embeddings (BERT also tested) fed to four ML classifiers combined by optimised weighted voting.
- **Findings:** Reports 99.91% accuracy, outperforming prior work on that dataset.
- **Limitations:** Single, old, saturated benchmark; paid embedding API; no out-of-distribution test.
- **Relevance:** Supports our use of **voting/stacking ensembles**, and is a good example of a ~99.9% result on a saturated benchmark — the pattern P4 warns about.

### P6. Wood, Kepkowski, Zinatullin, Darnley & Kaafar (2023) — Scam stages and scripts from scam-baiting calls
- **Venue:** arXiv:2307.01965, 2023. **Preprint.**
- **Problem:** Real scam-call recordings are scarce; can scripts and stages be extracted from public scam-baiting videos?
- **Dataset:** Transcripts of YouTube scam-baiting calls (~90 hours, primarily US scam types).
- **Method / algorithms:** Contextualised topic model + emotion recognition on scammer utterances; **a Hidden Markov Model over the topic sequence** to recover scam stages.
- **Findings:** Scam type can be identified from one utterance with 80–90% accuracy and from 5–6 utterances with 92–98%. The HMM captured a consistent scripted flow (described in coverage as introduction → assistance → threat → payment).
- **Limitations:** Scam-baiters are not real victims; small, US-centric sample.
- **Relevance:** **Direct justification for our HMM experiments.** They apply an HMM over *higher-level symbols* (topics) rather than raw words — the same design choice we adopt (lexicon categories as HMM observations).

### P7. Ai et al. (2024) — Defending against LLM-driven social engineering (SEConvo, ConvoSentinel)
- **Venue:** EMNLP 2024 (main conference), pp. 12880–12902. **Peer-reviewed.** DOI: 10.18653/v1/2024.emnlp-main.716 (arXiv:2406.12263)
- **Problem:** LLMs can generate chat-based social-engineering attacks; are LLMs good at detecting them?
- **Dataset:** SEConvo — LLM-simulated CSE conversations in academic and recruitment settings.
- **Method:** ConvoSentinel — a modular pipeline that flags malicious messages and aggregates them to a conversation-level verdict, with a retrieval-augmented module comparing messages to known attacks.
- **Findings:** Off-the-shelf LLMs generate convincing attacks but are sub-optimal detectors; the modular pipeline improves detection at message and conversation level at lower cost.
- **Limitations:** Synthetic LinkedIn-style scenarios; relies on LLM inference.
- **Relevance:** Supports our message-level indicator + conversation-level verdict structure, and our decision to run fully locally without paid APIs.

### P8. Salman, Ikram & Kaafar (2024) — Evasive techniques in SMS spam filtering
- **Venue:** IEEE Access, vol. 12, pp. 24306–24324, 2024. **Peer-reviewed.** DOI: 10.1109/ACCESS.2024.3364671
- **Problem:** How robust are SMS spam detectors and anti-spam services to spammer evasion?
- **Dataset:** A new, large public SMS spam dataset (released by the authors) plus longitudinal analysis.
- **Method / algorithms:** Shallow ML through deep neural networks on semantic and syntactic features; evasion experiments.
- **Findings:** Most shallow ML techniques and anti-spam services perform inadequately, and all evaluated approaches are susceptible to evasion strategies.
- **Limitations:** SMS rather than conversations.
- **Relevance:** Honest framing for our limitations section: a TF-IDF model can be evaded by rephrasing; our keyword indicators can be evaded by misspelling.

### P9. Dahiphale et al. — Google (2024) — LLM-powered scam detection for UPI / Google Pay
- **Venue:** arXiv:2410.19845, 2024. **Preprint.**
- **Problem:** Scam detection in UPI transactions in India, with explanations for human reviewers.
- **Method:** An LLM classifier plus a reasoning assistant that lists reasons for and against fraud.
- **Findings:** 93.33% accuracy on a *small* sample (40 transactions); recall above ~88–90%; precision around 30%; 89% of generated reasons matched reviewers or added valid insights.
- **Limitations:** Small evaluation sample (acknowledged by the authors); transaction data, not conversations; proprietary.
- **Relevance:** India/UPI context and a strong argument that **recall matters most** in scam screening — the same reason our risk thresholds favour catching scams over avoiding false alarms.

### P10. Ma et al. (2025) — TeleAntiFraud-28k: audio-text telecom fraud dataset
- **Venue:** ACM Multimedia 2025 (MM '25). **Peer-reviewed.** DOI: 10.1145/3746027.3755835 (arXiv:2503.24115)
- **Dataset:** 28,511 speech–text pairs built from ASR-transcribed, anonymised real calls, LLM self-instruct augmentation and multi-agent adversarial synthesis; annotated with step-by-step ("slow-thinking") reasoning.
- **Tasks:** Scenario classification, fraud detection, fraud-type classification.
- **Findings:** Untuned large audio-language models perform inadequately; fine-tuning Qwen2-Audio on the dataset gave a 27.5-point gain on TeleAntiFraud-Bench.
- **Limitations:** Chinese telecom context; large models.
- **Relevance:** Shows the field moving to multimodal/LLM models with *reasoning explanations*. Our project positions itself as the lightweight, transparent, laptop-runnable end of that spectrum.

### P11. Ahuja, Prateek & Singh (2026) — ICFD-31k: Indian conversational fraud dataset
- **Venue:** IJCAI 2026 (AI and Social Good track), pp. 7065–7073. **Peer-reviewed.** DOI: 10.24963/ijcai.2026/786
- **Problem:** No large public dataset for real-time conversational fraud detection in India.
- **Dataset:** ~31,000 *synthetic* English and Hinglish transcripts generated with Llama 3.3 70B from personas × 300+ scenarios; 10 fraud categories (bank & payment, e-commerce, utility, job, loan harassment, tech support, emergency/extortion, insurance/health, travel/lottery, government impersonation); chunk-level streaming labels and rationales.
- **Method / algorithms:** RoBERTa-base fine-tuned on full transcripts (M1) and on cumulative 3-second chunks (M2); Gemma-2B zero-shot baseline.
- **Findings:** M1 F1 99.40 in-domain but **92.97 on 5 unseen fraud types** (accuracy 87.20); streaming M2 F1 95.59; Gemma-2B zero-shot F1 35.83.
- **Limitations (stated by authors):** Synthetic, not validated on real recordings; moderate annotator agreement (κ = 0.534); in-domain score is an upper bound.
- **Relevance:** **The most directly comparable India-focused work.** It (a) shows our "India-focused" framing is a live research direction, (b) shows the same in-domain vs. cross-domain gap we measured, and (c) is the natural next dataset for our future work (restricted licence).

### P12. Lochstampfor & Roy (2026) — COVA: synthetic conversational smishing dataset
- **Venue:** arXiv:2604.11752, 2026. **Preprint.**
- **Dataset:** 3,201 multi-turn LLM-generated smishing conversations across eight elder-targeted scam categories.
- **Method / algorithms:** LR, Random Forest, Linear SVM, XGBoost (TF-IDF and engineered features) vs DistilBERT and Longformer.
- **Findings:** XGBoost + TF-IDF was best (72.5% accuracy, 0.691 macro-F1 on 3-class outcome prediction) and **TF-IDF models outperformed fine-tuned transformers**, attributed to input truncation and limited data.
- **Limitations:** Synthetic, English/US-centric; whole-conversation classification only.
- **Relevance:** Recent evidence that **TF-IDF + linear/ensemble models remain competitive** for conversation-level scam detection on modest data — the main justification for our feature choice.

### P13. Uddin, Mahiuddin & Sarker (2026) — Explainable transformer for phishing e-mail detection
- **Venue:** *Computer Networks* 277:112061, 2026. **Peer-reviewed.** DOI: 10.1016/j.comnet.2026.112061. (Earlier preprint, arXiv:2402.13871, 2024, used DistilBERT with LIME and Transformers-Interpret.)
- **Method:** Fine-tuned RoBERTa; explanations via LIME, Transformers-Interpret and a hybrid "LITA" method showing positive/negative word contributions.
- **Findings:** 98.45% accuracy, outperforming traditional ML baselines.
- **Limitations:** E-mail domain; post-hoc explanations of a black-box model.
- **Relevance:** Our explanation panel shows the same thing — positive and negative word contributions — but computed *exactly* from a linear model's coefficients rather than approximated post hoc.

---

## 2. Comparison table

| # | Paper (year) | Status | Domain / data | Model | Explain-ability | Out-of-domain test? | Key reported result |
|---|---|---|---|---|---|---|---|
| P1 | Derakhshan et al. (2021) | Peer-rev. (ACM IWSPA) | Phone scams, volunteer-written | Speech-act signatures + word embeddings | Signature matches | No | "High accuracy" |
| P2 | Tsinganos et al. (2022) | Peer-rev. (IEEE Access) | Chat social engineering | CNN + embeddings | Persuasion label | No | Persuasion recognised |
| P3 | Moussavou B. & Park (2023) | Peer-rev. (Mathematics) | Korean vishing transcripts | 1D CNN-BiLSTM-attention | Attention | No | Beats baselines on KorCCVi v2 |
| P4 | Jáñez-Martino et al. (2023) | Peer-rev. (AI Review) | Spam e-mail (review) | Various | — | **Yes (focus)** | Error up to 48.81% under shift |
| P5 | Ghourabi & Alohaly (2023) | Peer-rev. (Sensors) | SMS Spam Collection | GPT-3 emb. + weighted voting | No | No | 99.91% accuracy |
| P6 | Wood et al. (2023) | Preprint | Scam-baiting calls (US) | Topic model + **HMM** | Stage/topic | No | Scam type 92–98% (5–6 utt.) |
| P7 | Ai et al. (2024) | Peer-rev. (EMNLP) | LLM-generated chat SE | LLM + RAG pipeline | Message flags | Partly | LLMs sub-optimal detectors |
| P8 | Salman et al. (2024) | Peer-rev. (IEEE Access) | Large SMS dataset | Shallow ML → DNN | No | **Evasion tests** | All models evadable |
| P9 | Dahiphale et al. (2024) | Preprint | UPI / Google Pay (India) | LLM + reasoning | Reasons for/against | No | 93.33% acc. (n=40), recall ≈90% |
| P10 | Ma et al. (2025) | Peer-rev. (ACM MM) | Chinese telecom calls, audio+text | Qwen2-Audio fine-tune | Reasoning text | Bench | +27.5 points after tuning |
| P11 | Ahuja et al. (2026) | Peer-rev. (IJCAI) | **Indian** synthetic Eng./Hinglish | RoBERTa (static & streaming) | Rationales in data | **Yes** | F1 99.40 → 92.97 unseen types |
| P12 | Lochstampfor & Roy (2026) | Preprint | Synthetic smishing dialogues | **TF-IDF + LR/SVM/XGB** vs transformers | No | No | TF-IDF beats transformers |
| P13 | Uddin et al. (2026) | Peer-rev. (Computer Networks) | Phishing e-mail | RoBERTa + LIME/LITA | **Word contributions** | No | 98.45% accuracy |
| — | **This project** | Student project | Composite scam/non-scam transcripts + external English scripts | TF-IDF + LR / calibrated SVM / NB / ensembles | Exact linear contributions + rule indicators (kept separate) | **Yes** (external hold-out + shortcut audit) | See `reports/metrics.json` |

---

## 3. Research gap (scoped to the papers above)

What the reviewed literature shows:

1. **Benchmark scores are saturated but fragile.** In-domain results of 98–99.9% are common (P5, P11, P13), yet performance drops under dataset shift (P4), evasion (P8) and unseen scam types (P11: −6.43 F1). Few of the reviewed works test their model on a *differently sourced* dataset, and none of the reviewed conversation-level works audit their training data for **source shortcuts** (format, length, translated names).
2. **Explanations are either approximate or generated.** Recent systems explain via attention, LIME, or LLM-written rationales (P3, P9, P10, P11, P13). These explain a black box after the fact. Rule-based signatures (P1) are transparent but are not combined with a learned, *probability-calibrated* score.
3. **Risk scores are rarely calibrated.** The reviewed works report accuracy/precision/recall/F1. We did not find a reviewed conversation-scam paper that reports probability calibration (e.g., Brier score, reliability curve) even though a user-facing "risk %" implies one.
4. **India-focused conversational data is new and synthetic.** ICFD-31k (P11, 2026) describes itself as the first Indian conversational fraud dataset, and it is LLM-generated; P9 is transaction-level, not conversation-level.

**Gap statement (defensible):** *There is room for a lightweight, fully local, post-call system that (i) classifies whole conversations with classical, interpretable models, (ii) outputs a **calibrated** risk probability with documented thresholds, (iii) explains each prediction with **exact** model evidence kept separate from transparent rule-based indicators, and (iv) reports its **out-of-distribution** performance and dataset shortcuts honestly rather than only an in-domain accuracy.*

We do **not** claim to be the first scam-conversation classifier (P1, P3, P6, P11 precede us), nor to be India-specific: our training data contains very few India-specific terms (audit §5). India-relevant indicators (OTP, UPI, KYC, Aadhaar, remote-access apps) are covered by the transparent rule layer, and training on India-specific data such as ICFD-31k is future work.

## 4. Motivation → objectives

| Gap | Objective in this project | Where it is implemented |
|---|---|---|
| Fragile benchmark scores | Evaluate on an in-distribution test set **and** an external hold-out; quantify shortcuts | `src/scam_analyzer/train.py`, Model Comparison page, audit report |
| Approximate explanations | Exact per-word contributions from linear models; rule indicators shown separately | `explain.py`, `indicators.py`, Scam Detection page |
| Uncalibrated scores | Platt-calibrated SVM / LR probabilities, Brier score, reliability curve, configurable thresholds | `models.py`, `config/settings.toml` |
| Sequential structure | HMMs over interpretable symbol sequences (following P6), validated against labels | `hmm_analysis.py`, Advanced ML page |
| Practical deployment | Runs locally, no paid API, < 1 s per transcript | Streamlit app |

---

## 5. References (APA-style)

**Peer-reviewed**

1. Derakhshan, A., Harris, I. G., & Behzadi, M. (2021). Detecting telephone-based social engineering attacks using scam signatures. In *Proceedings of the 2021 ACM Workshop on Security and Privacy Analytics (IWSPA '21)* (pp. 67–73). ACM. https://doi.org/10.1145/3445970.3451152
2. Tsinganos, N., Mavridis, I., & Gritzalis, D. (2022). Utilizing convolutional neural networks and word embeddings for early-stage recognition of persuasion in chat-based social engineering attacks. *IEEE Access, 10*, 108517–108529. https://doi.org/10.1109/ACCESS.2022.3213681
3. Moussavou Boussougou, M. K., & Park, D.-J. (2023). Attention-based 1D CNN-BiLSTM hybrid model enhanced with FastText word embedding for Korean voice phishing detection. *Mathematics, 11*(14), 3217. https://doi.org/10.3390/math11143217
4. Jáñez-Martino, F., Alaiz-Rodríguez, R., González-Castro, V., Fidalgo, E., & Alegre, E. (2023). A review of spam email detection: Analysis of spammer strategies and the dataset shift problem. *Artificial Intelligence Review, 56*(2), 1145–1173. https://doi.org/10.1007/s10462-022-10195-4
5. Ghourabi, A., & Alohaly, M. (2023). Enhancing spam message classification and detection using transformer-based embedding and ensemble learning. *Sensors, 23*(8), 3861. https://doi.org/10.3390/s23083861
6. Ai, L., Kumarage, T. S., Bhattacharjee, A., Liu, Z., Hui, Z., Davinroy, M. S., Cook, J., Cassani, L., Trapeznikov, K., Kirchner, M., Basharat, A., Hoogs, A., Garland, J., Liu, H., & Hirschberg, J. (2024). Defending against social engineering attacks in the age of LLMs. In *Proceedings of EMNLP 2024* (pp. 12880–12902). ACL. https://doi.org/10.18653/v1/2024.emnlp-main.716
7. Salman, M., Ikram, M., & Kaafar, M. A. (2024). Investigating evasive techniques in SMS spam filtering: A comparative analysis of machine learning models. *IEEE Access, 12*, 24306–24324. https://doi.org/10.1109/ACCESS.2024.3364671
8. Ma, Z., Wang, P., Huang, M., Wang, J., Wu, K., Lv, X., Pang, Y., Yang, Y., Tang, W., & Kang, Y. (2025). TeleAntiFraud-28k: An audio-text slow-thinking dataset for telecom fraud detection. In *Proceedings of the 33rd ACM International Conference on Multimedia (MM '25)*. https://doi.org/10.1145/3746027.3755835
9. Ahuja, R., Prateek, K., & Singh, S. (2026). ICFD-31k: A large-scale dataset and benchmark for real-time conversational fraud detection. In *Proceedings of IJCAI 2026* (pp. 7065–7073). https://doi.org/10.24963/ijcai.2026/786
10. Uddin, M. A., Mahiuddin, M., & Sarker, I. H. (2026). An explainable transformer-based model for phishing email detection: A large language model approach. *Computer Networks, 277*, 112061. https://doi.org/10.1016/j.comnet.2026.112061

**Preprints (not peer-reviewed)**

11. Wood, I. D., Kepkowski, M., Zinatullin, L., Darnley, T., & Kaafar, M. A. (2023). An analysis of scam baiting calls: Identifying and extracting scam stages and scripts. *arXiv:2307.01965*. https://arxiv.org/abs/2307.01965
12. Dahiphale, D., Madiraju, N., Lin, J., Karve, R., Agrawal, M., Modwal, A., Balakrishnan, R., Shah, S., Kaushal, G., Mandawat, P., Hariramani, P., & Merchant, A. (2024). Enhancing trust and safety in digital payments: An LLM-powered approach. *arXiv:2410.19845*. https://arxiv.org/abs/2410.19845
13. Lochstampfor, C., & Roy, A. (2026). A synthetic conversational smishing dataset for social engineering detection. *arXiv:2604.11752*. https://arxiv.org/abs/2604.11752

**Background methods (older, cited for algorithms only)**

- Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In *Advances in Large Margin Classifiers*. MIT Press. — basis of `CalibratedClassifierCV(method="sigmoid")`.
- Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE, 77*(2), 257–286. https://doi.org/10.1109/5.18626 — Forward, Backward, Viterbi, Baum–Welch and scaling.
