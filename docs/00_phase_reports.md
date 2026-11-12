# Phase Completion Reports

## Phase 1 — Audit ✅
- **Completed:** inspected 3 notebooks, `eda.docx` (text + 7 figures), both PPT PDFs, all 5 data files. Reconstructed the missing cleaning script (exactly 30,409 rows). Reproduced reported accuracies within 0.2 pts. Measured class balance, exact and near-duplicates, test-set use in model selection, vectorizer fitting, format/length shortcut (92.8% with no words), out-of-domain drop (~82%), source-artifact tokens, and EDA/PPT contradictions.
- **Files:** `docs/01_audit_report.md`, `scripts/audit/*.py`.
- **Tests performed:** all audit numbers come from executed scripts.
- **Remaining issues:** GitHub repository URL not provided and not visible in the connected account, so the repository itself could not be audited.

## Phase 2 — Research ✅
- **Completed:** 13 papers (2021–2026) verified against publisher, ACL Anthology, IJCAI or arXiv pages; 10 peer-reviewed, 3 preprints, marked as such. Comparison table, scoped research gap, motivation → objectives, APA references, plain-language summaries.
- **Files:** `docs/02_literature_survey.md`.
- **Remaining issues:** where an abstract or proceedings page gave no number, none is quoted. Read the full PDFs of the papers you'll be questioned on most.

## Phase 3 — Architecture ✅
- **Completed:** package `src/scam_analyzer/` (single preprocessing path for training and inference), report-driven Streamlit app, config-driven thresholds, precomputed JSON reports.
- **Files:** see README §9.

## Phase 4 — Implementation ✅
- **Completed:** cleaning with URL/e-mail/phone/money/number/payment-app placeholders and source-name masking; leakage-safe splits; 8 models with 5-fold CV selection; Platt-calibrated SVM; exact linear explanations; 10-category rule layer with safe signals and rule escalation; EDA, clustering and HMM reports; from-scratch scaled Forward/Backward/Viterbi; 6-page dashboard; CLI fallback.
- **Files:** `src/scam_analyzer/*.py`, `app/**`, `scripts/*.py`, `config/settings.toml`, `models/*.joblib`, `reports/*.json`.
- **Key results:** default = calibrated Linear SVM, CV F1 99.44 ± 0.07; in-distribution test F1 99.43; external test accuracy 98.5; out-of-domain 82.2%.

## Phase 5 — Testing ✅
- **Tests performed (executed):** `pytest` → 50 passed, 1 expected failure (documented limitation). Covers preprocessing, rules, HMM algorithms vs `hmmlearn`, the 10 required cases, every model loading, and headless rendering of all 6 pages including scam, legitimate and empty-input flows. The real app was also run and screenshotted with headless Chromium (`docs/screenshots/`).
- **Issues found and fixed during testing:** phone regex missed spaced Indian numbers; "won" was treated as a currency; safe-signal lists ("OTP, PIN or password") only suppressed the first item; a mixed-type table column crashed the Dataset page; "100.0%" was displayed for calibrated estimates (now ">99.9%"); the HMM underflow demo used too short a sequence.
- **Remaining issue (documented, not hidden):** a genuine bank fraud-alert call is scored >99% scam (dataset bias + no negation handling). The rule layer handles it correctly.

## Phase 6 — Documentation ✅
- **Files:** `README.md` (results tables generated from `metrics.json`), `data/README.md`, `requirements.txt`, `.gitignore`, `pytest.ini`, `.streamlit/config.toml`, `docs/03_project_explained.md`.

## Phase 7 — Review preparation ✅
- **Files:** `docs/04_viva_question_bank.md` (61 Q&A), `docs/05_demo_script.md` (5-min script + backup plan), `docs/06_quick_revision_sheet.md`, plus 2-/5-minute literature talk tracks in `docs/07_literature_talk.md`.
