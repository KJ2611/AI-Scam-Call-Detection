"""Paths and settings shared by training, analysis and the app."""
from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
SETTINGS_FILE = ROOT / "config" / "settings.toml"

RANDOM_STATE = 42
TEST_SIZE = 0.20          # same 80:20 stratified split as the original notebooks
EXTERNAL_TRAIN_FRACTION = 0.50  # half of the English scripts are added to training, half held out


@lru_cache(maxsize=1)
def load_settings() -> dict:
    with open(SETTINGS_FILE, "rb") as fh:
        return tomllib.load(fh)


def ensure_dirs() -> None:
    for p in (DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, REPORTS_DIR / "figures"):
        p.mkdir(parents=True, exist_ok=True)
