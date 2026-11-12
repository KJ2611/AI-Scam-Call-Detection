"""Load the raw files, clean them, de-duplicate and build leakage-safe splits.

Splits produced (all saved to data/processed/):
* ``main_train`` / ``main_test`` — composite_train.csv after cleaning + de-duplication,
  80:20 stratified, random_state=42 (same protocol as the original notebooks).
* ``ext_train`` / ``ext_test`` — the 800 English_Scam / English_NonScam scripts (never used
  in the original notebooks), 50:50 stratified. ext_train is added to training; ext_test is a
  held-out *external* test set from a different source.
* ``composite_test_unseen`` — rows of composite_test.csv whose cleaned text is NOT in
  composite_train.csv (2,850 overlapping rows removed).
* ``fraud_call_unseen`` — rows of fraud_call.file not already contained in composite_train.csv.
"""
from __future__ import annotations

import re

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_PROCESSED, DATA_RAW, EXTERNAL_TRAIN_FRACTION, RANDOM_STATE, TEST_SIZE, ensure_dirs
from .preprocessing import clean_many

LABELS = {0: "Non-Scam", 1: "Scam"}


def _read_paragraph_file(path, strip_numbering: bool) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if strip_numbering:
        paras = [re.sub(r"^\d+\.\s*", "", p) for p in paras]
    return paras


def load_composite(name: str = "composite_train.csv") -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW / name)
    df = df.rename(columns={"text": "raw_text"})[["raw_text", "label"]]
    df["label"] = df["label"].astype(int)
    df["source"] = name.replace(".csv", "")
    return df


def load_english_scripts() -> pd.DataFrame:
    scam = _read_paragraph_file(DATA_RAW / "English_Scam.txt", strip_numbering=True)
    non = _read_paragraph_file(DATA_RAW / "English_NonScam.txt", strip_numbering=True)
    return pd.DataFrame({
        "raw_text": scam + non,
        "label": [1] * len(scam) + [0] * len(non),
        "source": "english_scripts",
    })


def load_fraud_call() -> pd.DataFrame:
    rows = []
    for line in (DATA_RAW / "fraud_call.file").read_text(encoding="utf-8", errors="replace").splitlines():
        if "\t" not in line:
            continue
        lab, txt = line.split("\t", 1)
        lab = lab.strip().lower()
        if lab in ("fraud", "normal"):
            rows.append((txt, 1 if lab == "fraud" else 0))
    df = pd.DataFrame(rows, columns=["raw_text", "label"])
    df["source"] = "fraud_call"
    return df


def clean_and_dedupe(df: pd.DataFrame, report: dict | None = None, name: str = "") -> pd.DataFrame:
    """Clean text, drop empties, drop texts that carry BOTH labels, drop duplicates."""
    out = df.copy()
    n0 = len(out)
    out["text"] = clean_many(out["raw_text"])
    out = out[out["text"].str.len() > 0]
    n_empty = n0 - len(out)
    labels_per_text = out.groupby("text")["label"].nunique()
    conflicting = set(labels_per_text[labels_per_text > 1].index)
    out = out[~out["text"].isin(conflicting)]
    n_before_dedup = len(out)
    out = out.drop_duplicates("text").reset_index(drop=True)
    if report is not None:
        report[name] = {
            "raw_rows": int(n0),
            "empty_after_cleaning": int(n_empty),
            "conflicting_label_texts_removed": int(len(conflicting)),
            "duplicates_removed": int(n_before_dedup - len(out)),
            "final_rows": int(len(out)),
            "scam": int(out["label"].sum()),
            "non_scam": int((out["label"] == 0).sum()),
        }
    return out


def build_datasets(save: bool = True) -> tuple[dict[str, pd.DataFrame], dict]:
    ensure_dirs()
    report: dict = {}
    comp = clean_and_dedupe(load_composite(), report, "composite_train")
    eng = clean_and_dedupe(load_english_scripts(), report, "english_scripts")
    fc = clean_and_dedupe(load_fraud_call(), report, "fraud_call")
    ct = clean_and_dedupe(load_composite("composite_test.csv"), report, "composite_test")

    train_texts = set(comp["text"])
    eng = eng[~eng["text"].isin(train_texts)]
    ct_unseen = ct[~ct["text"].isin(train_texts)].reset_index(drop=True)
    fc_unseen = fc[~fc["text"].isin(train_texts)].reset_index(drop=True)
    report["composite_test"]["overlap_with_composite_train"] = int(len(ct) - len(ct_unseen))
    report["fraud_call"]["overlap_with_composite_train"] = int(len(fc) - len(fc_unseen))

    main_train, main_test = train_test_split(
        comp, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=comp["label"])
    ext_train, ext_test = train_test_split(
        eng, test_size=1 - EXTERNAL_TRAIN_FRACTION, random_state=RANDOM_STATE, stratify=eng["label"])

    splits = {
        "main_train": main_train.reset_index(drop=True),
        "main_test": main_test.reset_index(drop=True),
        "ext_train": ext_train.reset_index(drop=True),
        "ext_test": ext_test.reset_index(drop=True),
        "english_all": eng.reset_index(drop=True),
        "composite_test_unseen": ct_unseen,
        "fraud_call_unseen": fc_unseen,
    }
    splits["train_full"] = pd.concat([splits["main_train"], splits["ext_train"]], ignore_index=True)
    report["splits"] = {k: {"rows": int(len(v)), "scam": int(v["label"].sum())} for k, v in splits.items()}
    if save:
        for k, v in splits.items():
            v.to_csv(DATA_PROCESSED / f"{k}.csv", index=False)
    return splits, report


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_PROCESSED / f"{name}.csv", keep_default_na=False)
