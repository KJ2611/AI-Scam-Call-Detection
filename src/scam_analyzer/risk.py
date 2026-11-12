"""Inference: transcript → prediction, calibrated probability, risk level, evidence, advice.

The output is a *model-assisted assessment*, not a definitive fraud determination.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from functools import lru_cache

import joblib
import numpy as np

from .config import MODELS_DIR, load_settings
from .explain import contributions, linear_weights
from .indicators import HIGH, detect_indicators
from .models import MODEL_INFO, positive_scores
from .preprocessing import PLACEHOLDERS, clean_text, split_utterances

DISCLAIMER = ("This is a model-assisted risk assessment based on the words in the transcript. "
              "It is not a definitive fraud determination; verify independently before acting.")

GENERAL_ADVICE = {
    "HIGH": [
        "Stop all contact with the caller. Do not call back on any number they gave you.",
        "If you shared an OTP, PIN, password or card details, call your bank on the number printed on your card and block the card / account now.",
        "If money was transferred, report immediately: in India call the 1930 cyber-crime helpline or file at cybercrime.gov.in (the first hours matter for freezing funds).",
        "Uninstall any app the caller asked you to install and change passwords from a different device.",
    ],
    "MEDIUM": [
        "Do not act on anything the caller asked for until you have verified it yourself.",
        "Contact the organisation using contact details from its official website or app — not those given on the call.",
        "Talk to someone you trust before sharing information or paying.",
    ],
    "LOW": [
        "No strong scam signal was found, but stay alert: never share OTPs, PINs or passwords on a call.",
        "If anything in the call felt off, verify through the organisation's official channel.",
    ],
}


@dataclass
class AnalyzerConfig:
    medium: float
    high: float
    decision: float
    rule_escalation: bool
    escalate_min_high: int
    min_words: int
    max_chars: int
    top_k: int

    @classmethod
    def from_settings(cls) -> "AnalyzerConfig":
        s = load_settings()
        r, a = s["risk"], s["app"]
        return cls(r["medium_threshold"], r["high_threshold"], r["decision_threshold"], r["rule_escalation"],
                   r["escalate_min_high_severity"], a["min_words"], a["max_chars"], a["top_k_evidence"])


def risk_level(p: float, cfg: AnalyzerConfig) -> str:
    if p >= cfg.high:
        return "HIGH"
    if p >= cfg.medium:
        return "MEDIUM"
    return "LOW"


def text_stats(raw: str, cleaned: str) -> dict:
    words = re.findall(r"[A-Za-z']+", raw)
    toks = cleaned.split()
    utt = split_utterances(raw)
    return {
        "words": len(words),
        "characters": len(raw),
        "unique_words": len({w.lower() for w in words}),
        "lexical_diversity": round(len({w.lower() for w in words}) / max(len(words), 1), 3),
        "avg_word_length": round(sum(map(len, words)) / max(len(words), 1), 2),
        "sentences_or_turns": len(utt),
        "questions": raw.count("?"),
        "exclamations": raw.count("!"),
        "uppercase_ratio": round(sum(c.isupper() for c in raw) / max(sum(c.isalpha() for c in raw), 1), 3),
        "numbers": toks.count("__num__"),
        "phone_numbers": toks.count("__phone__"),
        "links": toks.count("__url__"),
        "money_mentions": toks.count("__money__"),
        "email_addresses": toks.count("__email__"),
    }


class ScamAnalyzer:
    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = models_dir
        self.vectorizer = joblib.load(models_dir / "tfidf_vectorizer.joblib")
        meta = joblib.load(models_dir / "meta.joblib")
        self.best_model = meta["best_model"]
        self.feature_names = np.asarray(meta["feature_names"])
        self._models: dict = {}
        self.cfg = AnalyzerConfig.from_settings()

    # ------------------------------------------------------------------ models
    def available_models(self) -> list[str]:
        return [k for k in MODEL_INFO if (self.models_dir / f"{k}.joblib").exists()]

    def model(self, key: str):
        if key not in self._models:
            self._models[key] = joblib.load(self.models_dir / f"{key}.joblib")
        return self._models[key]

    def _score(self, key: str, X) -> tuple[float, int]:
        m = self.model(key)
        s = float(positive_scores(m, X)[0])
        pred = int(m.predict(X)[0]) if key == "hard_voting" else int(s >= self.cfg.decision)
        return s, pred

    # ------------------------------------------------------------------ main API
    def analyze(self, text: str, model_key: str | None = None, compare_all: bool = False) -> dict:
        t0 = time.perf_counter()
        key = model_key if model_key and model_key != "auto" else self.best_model
        raw = "" if text is None else str(text)
        warnings: list[str] = []
        if not raw.strip():
            return {"ok": False, "error": "Please paste a conversation transcript first."}
        truncated = False
        if len(raw) > self.cfg.max_chars:
            raw = raw[: self.cfg.max_chars]
            truncated = True
            warnings.append(f"Input was longer than {self.cfg.max_chars:,} characters; only the first part was analysed.")
        cleaned = clean_text(raw)
        if not cleaned:
            return {"ok": False, "error": "The text contains no analysable words (only symbols or whitespace)."}
        X = self.vectorizer.transform([cleaned])
        n_words = len(re.findall(r"[A-Za-z']+", raw))
        if n_words < self.cfg.min_words:
            warnings.append(f"Very short input ({n_words} words). The model has little evidence; treat the result as unreliable.")
        if X.nnz == 0:
            warnings.append("None of the words appear in the model's vocabulary, so the model has no evidence and returns its base rate. "
                            "The input may be in another language or unrelated to phone conversations.")
        elif X.nnz < 5:
            warnings.append(f"Only {X.nnz} known terms were recognised — low evidence.")

        info = MODEL_INFO[key]
        score, pred = self._score(key, X)
        model_risk = risk_level(score, self.cfg) if info["probabilistic"] else ("HIGH" if score >= 2 / 3 else "MEDIUM" if score > 0 else "LOW")

        ind = detect_indicators(raw)
        final_risk, escalated = model_risk, None
        if self.cfg.rule_escalation and model_risk == "LOW" and ind["n_high"] >= self.cfg.escalate_min_high:
            final_risk = "MEDIUM"
            escalated = ("Raised from LOW to MEDIUM by the rule layer: the transcript contains "
                         + ", ".join(d["label"] for d in ind["indicators"] if d["severity"] == HIGH)
                         + ". The model probability itself was not changed.")

        # model evidence
        lw = linear_weights(key, self.model(key))
        evidence_from = info["name"]
        if lw is None:
            lw = linear_weights("logistic_regression", self.model("logistic_regression"))
            evidence_from = "Logistic Regression (this ensemble has no exact per-word decomposition)"
        ev = contributions(X, lw[0], self.feature_names, self.cfg.top_k)
        ev["units"], ev["source"] = lw[2], evidence_from

        all_scores = None
        if compare_all:
            all_scores = {}
            for k in self.available_models():
                s, p = self._score(k, X)
                all_scores[k] = {"name": MODEL_INFO[k]["name"], "score": s, "prediction": p,
                                 "probabilistic": MODEL_INFO[k]["probabilistic"]}

        return {
            "ok": True,
            "model_key": key, "model_name": info["name"], "probabilistic": info["probabilistic"],
            "score": score, "prediction": "SCAM" if pred == 1 else "NON-SCAM",
            "model_risk": model_risk, "risk": final_risk, "escalation": escalated,
            "uncertain": info["probabilistic"] and self.cfg.medium <= score < self.cfg.high,
            "thresholds": {"medium": self.cfg.medium, "high": self.cfg.high, "decision": self.cfg.decision},
            "indicators": ind, "evidence": ev,
            "explanation": self._explain(score, final_risk, ev, ind, info),
            "recommendations": self._advice(final_risk, ind),
            "stats": text_stats(raw, cleaned), "known_terms": int(X.nnz),
            "cleaned_preview": " ".join(PLACEHOLDERS.get(t, t) for t in cleaned.split()[:120]),
            "warnings": warnings, "truncated": truncated, "all_models": all_scores,
            "disclaimer": DISCLAIMER, "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    # ------------------------------------------------------------------ text
    @staticmethod
    def _explain(score, risk, ev, ind, info) -> list[str]:
        lines = []
        shown = ">99.9%" if score >= 0.999 else "<0.1%" if score <= 0.001 else f"{score:.1%}"
        what = f"{shown} estimated probability of scam" if info["probabilistic"] else f"{score:.0%} of the voting models said scam"
        lines.append(f"**Model assessment** — {info['name']}: {what} → **{risk}** risk.")
        top = [t for t, *_ in ev["toward_scam"][:5]]
        safe = [t for t, *_ in ev["toward_non_scam"][:4]]
        if top:
            lines.append("Words/phrases that pushed the model **toward scam**: " + ", ".join(f"“{t}”" for t in top) + ".")
        if safe:
            lines.append("Words/phrases that pushed it **toward legitimate**: " + ", ".join(f"“{t}”" for t in safe) + ".")
        if ind["indicators"]:
            hi = [d["label"] for d in ind["indicators"] if d["severity"] == HIGH]
            md = [d["label"] for d in ind["indicators"] if d["severity"] != HIGH]
            if hi:
                lines.append("**Rule-based red flags (high severity):** " + "; ".join(hi) + ".")
            if md:
                lines.append("**Rule-based manipulation tactics:** " + "; ".join(md) + ".")
        else:
            lines.append("No rule-based red-flag phrases were found.")
        if ind["safe_signals"]:
            lines.append("Reassuring phrases typical of genuine organisations were also found: "
                         + ", ".join(f"“{s}”" for s in ind["safe_signals"][:3]) + ".")
        lines.append("_The model reacts to word patterns learned from the training data; it does not understand intent. "
                     "Rule flags are keyword matches, not proof of fraud._")
        return lines

    @staticmethod
    def _advice(risk: str, ind: dict) -> list[str]:
        tips = list(GENERAL_ADVICE[risk])
        for d in ind["indicators"]:
            if d["advice"] not in tips:
                tips.append(d["advice"])
        return tips[:8]


@lru_cache(maxsize=1)
def get_analyzer() -> ScamAnalyzer:
    return ScamAnalyzer()
