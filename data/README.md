# Data setup

The raw files are **not committed** to Git (size and third-party licensing). Put them here:

```
data/raw/
├── composite_train.csv      # columns: text,label   (1 = scam, 0 = non-scam)
├── composite_test.csv       # same format
├── English_Scam.txt         # 400 numbered scam scripts, blank-line separated
├── English_NonScam.txt      # 400 non-scam scripts, blank-line separated
└── fraud_call.file          # tab-separated: <fraud|normal>\t<text>
```

They come from the project archive `project(2).zip` (folders `archive (1)/` and `archive/`). Extract and flatten:

```bash
unzip "project(2).zip" -d /tmp/proj
cp "/tmp/proj/project/archive (1)/"*.csv /tmp/proj/project/archive/*.txt /tmp/proj/project/fraud_call.file data/raw/
```

`python scripts/train_models.py` then writes the cleaned, de-duplicated splits to `data/processed/`:

| File | Rows | Purpose |
|---|---|---|
| `main_train.csv` | 24,313 | composite 80% (training) |
| `main_test.csv` | 6,079 | composite 20% (in-distribution test) |
| `ext_train.csv` | 397 | English scripts, 50% (added to training) |
| `ext_test.csv` | 397 | English scripts, 50% (external test) |
| `train_full.csv` | 24,710 | main_train + ext_train |
| `english_all.csv` | 794 | all English scripts (out-of-domain ablation) |
| `composite_test_unseen.csv` | 5,887 | composite_test rows not in composite_train |
| `fraud_call_unseen.csv` | 422 | fraud_call rows not in composite_train |

Displayed samples in the app mask phone numbers, e-mail addresses and links.
