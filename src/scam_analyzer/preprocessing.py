"""Text preprocessing — ONE function used for training and for live inference.

Design notes (see docs/01_audit_report.md §2 and §5):
* The original notebooks lower-cased text, replaced digits with ``<number>`` and removed
  punctuation. The slides also promised ``<PHONE>`` and URL handling, which the original
  cleaning did not do. This module implements all of them.
* Placeholders are written as ``__url__``-style tokens so that scikit-learn's default
  TF-IDF tokenizer (``\\b\\w\\w+\\b``) keeps them as single features.
* Speaker-name tokens that identify the *source* of the scam dialogues (e.g. translated
  names such as "zhang", "xiao") are removed, because in this dataset they occur only in
  scam rows and let a model shortcut the task (audit §5).
"""
from __future__ import annotations

import re
from typing import Iterable

PLACEHOLDERS = {
    "__url__": "<URL>",
    "__email__": "<EMAIL>",
    "__phone__": "<PHONE>",
    "__money__": "<MONEY>",
    "__num__": "<NUMBER>",
    "__payapp__": "<PAYMENT_APP>",
}

# Tokens that identify the data source rather than scam behaviour (audit §5).
SOURCE_ARTIFACT_TOKENS = frozenset(
    """li zhang xiao ming wei hua wang pao jianguo lao liu chen zhao zhou huang lin yang
    pudong xiaoming xiaoli xiaozhang laowang""".split()
)

PAYMENT_APPS = r"wechat|alipay|paytm|phonepe|gpay|google\s?pay|bhim|venmo|zelle|cash\s?app|paypal"
CURRENCY_WORDS = r"rupees?|rs|inr|dollars?|usd|pounds?|gbp|euros?|yuan|rmb|lakhs?|crores?"  # not "won": verb

_RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_RE_URL = re.compile(r"(?:https?://|www\.)\S+|\b[\w-]+\.(?:com|net|org|in|co\.uk|co|info|biz|xyz|ly|io)(?:/\S*)?\b", re.I)
_RE_PHONE_CANDIDATE = re.compile(r"(?<![\w+])\+?\(?\d[\d\s\-().]{7,18}\d(?!\w)")
_RE_MONEY_SYMBOL = re.compile(r"(?:[₹$£€¥]|\brs\.?|\binr)\s?\d[\d,]*(?:\.\d+)?", re.I)
_RE_PAYAPP = re.compile(rf"\b(?:{PAYMENT_APPS})\b", re.I)
_RE_CURRENCY = re.compile(rf"\b(?:{CURRENCY_WORDS})\b", re.I)
_RE_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_RE_NON_TEXT = re.compile(r"[^a-z_\s]")
_RE_SPACES = re.compile(r"\s+")


def _phone_or_keep(m: re.Match) -> str:
    """A digit run (allowing spaces/dashes/brackets) with 10-13 digits is treated as a phone number."""
    n_digits = sum(ch.isdigit() for ch in m.group(0))
    return " __phone__ " if 10 <= n_digits <= 13 else m.group(0)


def clean_text(text: object, mask_source_artifacts: bool = True) -> str:
    """Normalise a raw transcript into the form the models were trained on."""
    if text is None:
        return ""
    t = str(text)
    if not t.strip():
        return ""
    t = _RE_EMAIL.sub(" __email__ ", t)
    t = _RE_URL.sub(" __url__ ", t)
    t = _RE_MONEY_SYMBOL.sub(" __money__ ", t)
    t = _RE_PHONE_CANDIDATE.sub(_phone_or_keep, t)
    t = t.lower()
    t = _RE_PAYAPP.sub(" __payapp__ ", t)
    t = _RE_CURRENCY.sub(" __money__ ", t)
    t = _RE_NUMBER.sub(" __num__ ", t)
    t = t.replace("'", "")                    # don't -> dont, keeps contractions as one token
    t = _RE_NON_TEXT.sub(" ", t)
    # underscores are only legal inside placeholders
    tokens = []
    for tok in t.split():
        if "_" in tok and tok not in PLACEHOLDERS:
            tok = tok.replace("_", " ").strip()
            if not tok:
                continue
        if mask_source_artifacts and tok in SOURCE_ARTIFACT_TOKENS:
            continue
        tokens.append(tok)
    return _RE_SPACES.sub(" ", " ".join(tokens)).strip()


def clean_many(texts: Iterable[object], mask_source_artifacts: bool = True) -> list[str]:
    return [clean_text(t, mask_source_artifacts) for t in texts]


def legacy_clean_text(text: object) -> str:
    """Reconstruction of the ORIGINAL notebooks' cleaning (reproduces 30,409 rows).

    Kept only to reproduce the historical numbers in the audit; not used by the app.
    """
    t = str(text).lower()
    t = re.sub(r"\d+", "<number>", t)
    t = re.sub(r"[^a-z<>\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def pretty_token(token: str) -> str:
    """Render placeholders as <URL>, <PHONE> … for display."""
    return " ".join(PLACEHOLDERS.get(p, p) for p in token.split())


def split_utterances(raw: str) -> list[str]:
    """Best-effort split of a transcript into utterances (for display and HMM symbols)."""
    raw = str(raw).strip()
    # Python-list style:  'Hello ...', "Yes ...", ...
    parts = re.split(r"""['"]\s*,\s*['"]""", raw)
    if len(parts) > 1:
        return [p.strip(" '\"\n") for p in parts if p.strip(" '\"\n")]
    lines = [l.strip() for l in re.split(r"\n+", raw) if l.strip()]
    if len(lines) > 1:
        return lines
    sents = re.split(r"(?<=[.!?])\s+", raw)
    return [s.strip() for s in sents if s.strip()]
