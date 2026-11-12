"""Page 6 — About the project."""
from __future__ import annotations

import streamlit as st

from ui import load_json, pct


def render():
    st.title("About the project")
    m = load_json("metrics.json") or {}
    best = m.get("models", {}).get(m.get("best_model", ""), {})
    t = st.tabs(["Problem & objectives", "Methodology", "Data", "Architecture & stack", "Limitations & future work", "References"])
    with t[0]:
        st.markdown("""
**Problem.** Scam calls use impersonation, urgency and manipulation. People often realise a call was suspicious only afterwards, and
reading a long transcript for warning signs is slow. Keyword lists alone miss context and raise false alarms.

**Objectives**
1. *Scam classification* — classify a conversation as scam / non-scam and compare several ML models fairly.
2. *Pattern identification* — surface the suspicious words, phrases and manipulation tactics, and analyse how scam and legitimate language differ.
3. *Risk assessment* — give a calibrated probability, a Low / Medium / High risk level with documented thresholds, an explanation and safety advice.

**Research gap addressed** (see `docs/02_literature_survey.md`): recent work reaches 98–99% on in-domain benchmarks with deep or LLM models but rarely
(a) reports calibrated probabilities, (b) separates exact model evidence from transparent rules, or (c) audits datasets for shortcuts and reports
out-of-distribution results. This project does all three with lightweight models that run on a laptop.
""")
    with t[1]:
        st.markdown(f"""
1. **Audit** of the original notebooks: results reproduced, leakage and dataset shortcuts measured (`docs/01_audit_report.md`).
2. **Preprocessing** — one function for training and inference: lower-casing; `<URL>`, `<EMAIL>`, `<PHONE>`, `<MONEY>`, `<NUMBER>`, `<PAYMENT_APP>`
   placeholders; punctuation removed; source-identifying names masked; empty, duplicate and conflicting-label rows dropped.
3. **Features** — TF-IDF, 10,000 uni+bigrams, fitted on the training split only.
4. **Models** — Logistic Regression, Multinomial NB, Linear SVM (Platt-calibrated), hard & soft voting, stacking, bagging, AdaBoost.
5. **Selection** — 5-fold stratified cross-validation on the training split (TF-IDF refitted inside each fold); metric: F1 of the scam class.
6. **Evaluation** — once, on four held-out sets; accuracy, precision, recall, F1, confusion matrix, ROC-AUC, average precision, Brier score, ECE.
7. **Risk** — calibrated P(scam) → LOW < 0.35 ≤ MEDIUM < 0.70 ≤ HIGH (configurable). A HIGH-severity rule match can raise LOW to MEDIUM; the probability itself is never altered.
8. **Explanation** — for linear models every term's contribution x·w is exact and additive; rule indicators are shown separately and labelled as such.
9. **Supplementary** — clustering and class-conditional HMMs, with from-scratch Forward / Backward / Viterbi verified against `hmmlearn`.
""")
        if best:
            st.markdown(f"**Current default model:** {best['info']['name']} — CV F1 {pct(best['cv']['f1']['mean'], 2)}, in-distribution test F1 "
                        f"{pct(best['test']['main_test']['f1'], 2)}, external test accuracy {pct(best['test']['ext_test']['accuracy'])}.")
    with t[2]:
        st.markdown("""
| Source | Use | Notes |
|---|---|---|
| `composite_train.csv` | Train (80%) / test (20%) | 36,751 raw rows → ~30.4k unique after cleaning; ~64% scam; mixes translated scam dialogues, call transcripts and SMS |
| `English_Scam.txt`, `English_NonScam.txt` | 50% added to training, 50% external test | 400 + 400 hand-written template scripts; never used by the original notebooks |
| `composite_test.csv` | Extra in-distribution test | Rows already present in the training file are removed first |
| `fraud_call.file` | Extra SMS-style test | ~92% of its rows are already inside `composite_train.csv`; only the rest are used |

Sensitive details in displayed samples (phone numbers, e-mails, links) are masked.
""")
    with t[3]:
        st.markdown("""
```
data/raw ──► src/scam_analyzer/data.py (clean, dedupe, split) ──► data/processed/*.csv
                         │
                         ▼
          src/scam_analyzer/train.py ──► models/*.joblib + reports/metrics.json
          src/scam_analyzer/eda.py   ──► reports/eda.json
          src/scam_analyzer/advanced.py ─► reports/advanced.json + models/hmm_models.joblib
                         │
                         ▼
 app/streamlit_app.py ─► risk.ScamAnalyzer: clean → TF-IDF → model → calibrated P(scam)
                         + explain.py (exact contributions) + indicators.py (rules) → UI
```
**Stack:** Python 3.11+, pandas, NumPy, scikit-learn, hmmlearn, joblib, Streamlit, Plotly. No paid APIs; inference runs offline in a few milliseconds.
""")
    with t[4]:
        st.markdown("""
**Limitations**
- **Dataset shortcuts:** in the composite data, format, length, topic and translated names predict the label; in-distribution accuracy is optimistic.
- **Bag-of-words model:** TF-IDF sees which words occur, not who said them or why. It does not understand intent, sarcasm or negation
  ("I will *never* ask for your OTP" contains "OTP"). The rule layer handles a few safe phrasings explicitly.
- **Limited Indian context:** few OTP/UPI/KYC/Aadhaar examples in training; Hinglish and regional languages are not supported.
- **Text only:** no audio, caller ID, or speaker turns; transcription errors are not modelled.
- **Evasion:** rephrasing or misspelling can lower the score; keyword rules can be dodged.
- **Calibration is dataset-specific:** probabilities are calibrated to the training mix (~64% scam), not to real-world scam prevalence.

**Future work**
- Train and evaluate on India-focused conversational data (e.g. ICFD-31k, English + Hinglish) and real, consented transcripts.
- Speaker-aware features (who asks for what) and turn-level / streaming detection during the call.
- Compare against a fine-tuned multilingual transformer with the same calibration and external-test protocol.
- Recalibrate for realistic prevalence; collect user feedback for active learning.
""")
    with t[5]:
        st.markdown("""
See `docs/02_literature_survey.md` for the full survey (13 verified papers, 2021–2026, peer-reviewed vs preprint marked). Key works:
Derakhshan et al. 2021 (ACM IWSPA); Tsinganos et al. 2022 (IEEE Access); Moussavou Boussougou & Park 2023 (Mathematics); Jáñez-Martino et al. 2023
(AI Review); Ghourabi & Alohaly 2023 (Sensors); Wood et al. 2023 (arXiv); Ai et al. 2024 (EMNLP); Salman et al. 2024 (IEEE Access);
Dahiphale et al. 2024 (arXiv); Ma et al. 2025 (ACM MM); Ahuja et al. 2026 (IJCAI); Lochstampfor & Roy 2026 (arXiv); Uddin et al. 2026 (Computer Networks).
""")
