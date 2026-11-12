# 5-Minute Live Demo Script

**Before the panel arrives (5 min earlier)**
1. `cd scam-risk-analyzer && source .venv/bin/activate` (or your venv)
2. `streamlit run app/streamlit_app.py` → browser opens at http://localhost:8501
3. Open the **Scam detection** page once and click *Analyze* on any example (loads the models into cache so the live click is instant).
4. Go back to **Home**. Keep `docs/` open in a second window. Set browser zoom to 110–125% for the projector.
5. Have the backup ready (see the end).

---

### 0:00–0:40 — Home page
- **Say:** "This is a post-call scam risk analyzer. You paste a transcript; it gives a calibrated scam probability, explains which words drove it, flags manipulation tactics, and gives advice. Everything runs locally."
- **Point at** the six-step pipeline: transcript → preprocessing → TF-IDF → classification → risk → explanation.
- **Point at** the numbers: "In-distribution F1 is about 99%, but notice the external-test and out-of-domain numbers — I'll explain why we report those too."

### 0:40–2:30 — Scam detection (the main event)
1. Click **Scam sample** (bank impersonation + OTP). Click **Analyze conversation**.
   - **Show:** big SCAM label, HIGH risk, probability meter with the 35% / 70% thresholds.
   - **Say:** "The probability comes from a Platt-calibrated linear SVM — we don't turn the raw SVM margin into a fake percentage."
2. **Why** tab: "These bars are exact contributions — TF-IDF value times the learned weight — so they add up to the model's score. Red pushes toward scam, green toward legitimate."
3. **Red flags & advice** tab: "Separately, a transparent rule layer found an OTP request, account-blocking threat, impersonation, urgency and secrecy. These are keyword patterns — reasons to look closer, not proof. Advice includes India's 1930 cyber-fraud helpline."
4. **Highlighted transcript** tab: red = rule flags, cyan = model scam terms. Scroll to the HMM stage plot: "Supplementary: hidden stages under the scam HMM."
5. Click **Legit sample** → Analyze. **Show:** NON-SCAM, LOW risk. "Appointment reminder — no red flags."
6. *(Shows honesty — recommended)* Choose **Genuine bank fraud-alert call** → Analyze. The model says **SCAM, >99%**. **Say:** "This is a known failure. A real bank calls and says 'we will never ask for your OTP'. In our training data nearly every bank-card conversation is a scam, and a bag-of-words model can't read negation. Look at the rule layer though: no credential *request*, and three safe signals. We documented this rather than hiding it — a rule that lowers risk when it hears 'we never ask for OTP' would be easy for scammers to exploit."
7. *(If time)* **Ambiguous — job offer** → shows MEDIUM / uncertain band, with the 'registration fee' flag. "Borderline cases are shown as uncertain, not forced to a confident answer."

### 2:30–3:40 — Model comparison
- **Table:** eight models; ★ marks the default, chosen by 5-fold CV F1 on training data — "the test set did not choose the model."
- **Confusion matrices:** "False negatives are the dangerous error — a scam called safe."
- **Reliability plot:** "Calibrated SVM and LR lie near the diagonal; Naive Bayes is over-confident."
- Scroll to **Why the scores look the way they do**:
  - "We reproduced the notebook results within 0.2 points."
  - "But a model that reads *no words* — only length and format — scores 92.8%. The dataset has shortcuts."
  - "Trained without the English scripts, accuracy on them drops to 82.2%. That's the honest out-of-domain number."

### 3:40–4:30 — Dataset explorer & Advanced ML (pick one or two)
- **Dataset → Length tab:** "Earlier EDA said scams are 'more concise' using means; the medians say the opposite. We corrected that."
- **Dataset → Sources tab:** artifact-token table — "names like 'zhang' appear only in scam rows, so we mask them."
- **Advanced → Clustering:** "Clusters are topics, not labels — ARI is low and we say so." K-Means k=8 topic table.
- **Advanced → HMM:** transition matrix; "Forward/Viterbi implemented from scratch and matched against hmmlearn."

### 4:30–5:00 — Conclusion
"To summarise: a working, locally-run analyzer that classifies conversations, gives a calibrated risk with documented thresholds, explains itself with exact model evidence plus transparent rules, and reports its out-of-distribution performance honestly. Next steps are India-focused training data such as ICFD-31k and speaker-aware, real-time detection."

---

## Backup plan
| Problem | What to do |
|---|---|
| **No internet** | Nothing changes — the app is fully offline. (Only the web font falls back to the system font.) |
| App won't start / dependency error | Run `python scripts/demo_cli.py "paste transcript"` — prints the same verdict, probability, evidence and red flags in the terminal. |
| Models missing on the demo laptop | `python scripts/train_models.py --fast` (~10 min; skips CV for the two slow tree ensembles) then `python scripts/run_analysis.py`. Do this the day before. |
| Laptop fails entirely | Show `docs/screenshots/*.png` and the README results tables; walk through `reports/metrics.json`. |
| Port 8501 busy | `streamlit run app/streamlit_app.py --server.port 8502` |
| Slow first analysis | That is model loading; click once before the panel arrives (step 3 above). |

## Transcripts to keep handy (copy-paste)
- Scam (OTP): *"Hello, I am from your bank. Your account will be blocked today. To complete KYC verification please tell me the OTP sent to your phone right now."*
- Legit: *"Hi, this is the dental clinic reminding you of your appointment tomorrow at 4:30. Reply if you need to reschedule. No payment is needed now."*
- Tricky legit: *"This is your bank's fraud team. Was the 2,300 rupee card payment yours? Remember we will never ask for your OTP or PIN."*
