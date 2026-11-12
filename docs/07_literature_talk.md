# Literature Survey — Talk Tracks

## 2-minute version
"Detecting scams from what is *said* is an established problem. Early work treated it like signature matching: Derakhshan and colleagues in 2021 described each phone scam as a set of speech acts — 'ask for SSN', 'threaten arrest' — and searched calls for them. Tsinganos and colleagues (IEEE Access 2022) trained a CNN to spot persuasion in chat-based social engineering.

From 2023 the field moved to deep models on call transcripts — for example a CNN-BiLSTM with attention for Korean voice phishing — and then to large datasets built with LLMs: TeleAntiFraud-28k at ACM Multimedia 2025, ConvoSentinel at EMNLP 2024, and, closest to us, ICFD-31k at IJCAI 2026, the first Indian conversational fraud dataset in English and Hinglish.

Two findings matter for our project. First, in-domain scores are near-perfect everywhere, but they don't transfer: a review in *AI Review* 2023 showed error rates up to 49% under dataset shift, and ICFD-31k's RoBERTa drops from 99.4 to 93.0 F1 on unseen scam types. Second, explanations are mostly post-hoc (LIME, attention) or LLM-written, and calibration is rarely reported.

Our gap: a lightweight, local system that gives a calibrated risk, exact explanations kept separate from transparent rules, and honest out-of-distribution results. We don't claim to be the first scam classifier — we claim this combination."

## 5-minute version (add to the above)
1. **Where TF-IDF still wins.** Lochstampfor & Roy (2026 preprint) found TF-IDF + XGBoost beat fine-tuned DistilBERT/Longformer on 3,201 conversational smishing dialogues — limited data and input truncation hurt transformers. That supports our choice of TF-IDF + linear models for ~25k conversations on a laptop.
2. **Ensembles in the literature.** Ghourabi & Alohaly (Sensors 2023) used GPT-3 embeddings with weighted voting and reached 99.91% on the SMS Spam Collection — a saturated benchmark. It motivates our voting/stacking comparison and shows why we distrust single near-100% numbers.
3. **Robustness.** Salman, Ikram & Kaafar (IEEE Access 2024): all ML spam filters they tested could be evaded. Our limitation section says the same about rephrasing.
4. **Sequential structure.** Wood et al. (2023 preprint) applied an HMM over topic sequences from scam-baiting calls and found scripted stages (introduction, assistance, threat, payment). That is why our HMM uses higher-level stage symbols, not raw words.
5. **India.** Dahiphale et al. (Google, 2024 preprint) used LLMs for UPI scam review, reporting high recall but ~30% precision on a small sample — another reason we favour recall. ICFD-31k is synthetic (Llama 3.3) with moderate annotator agreement (κ = 0.534); our India-specific cues therefore sit in a transparent rule layer, and training on ICFD-31k is future work.
6. **Explainability.** Uddin, Mahiuddin & Sarker (Computer Networks 2026) explain a RoBERTa phishing model with LIME and a hybrid "LITA". Our explanations are exact for the linear model — contributions sum to the score — and the rule evidence is labelled separately.

## Likely literature questions
- **"Which paper is closest to yours?"** ICFD-31k (Ahuja et al., IJCAI 2026): conversation-level, India-focused, reports in-domain vs cross-domain. Differences: theirs is RoBERTa on synthetic English/Hinglish data with streaming labels; ours is a calibrated linear model with exact explanations and a shortcut audit on a different dataset.
- **"Why not use a transformer like most recent papers?"** See P12 (TF-IDF competitive on modest data), plus exactness of explanations and offline speed. A calibrated transformer under the same evaluation is future work.
- **"Which of your papers are not peer-reviewed?"** Wood et al. 2023, Dahiphale et al. 2024, Lochstampfor & Roy 2026 (arXiv). Uddin et al. was a 2024 preprint, now published in Computer Networks (2026).
- **"What does dataset shift mean?"** Training and deployment data come from different distributions — covariate shift (features), prior shift (class balance), concept shift (the feature–label relation). Jáñez-Martino et al. 2023 is the reference; our out-of-domain test measures it directly.
- **"Is your research gap real or invented?"** It is scoped to the reviewed papers and stated as a combination, not a first. Each component has precedent; the combination plus the honest evaluation is what we contribute.
