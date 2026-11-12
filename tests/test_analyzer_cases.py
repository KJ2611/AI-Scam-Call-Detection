"""The 10 required end-to-end test cases, run against the trained models.

Skipped automatically if models have not been trained yet.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from examples import EXAMPLES  # noqa: E402

from scam_analyzer.config import MODELS_DIR  # noqa: E402

pytestmark = pytest.mark.skipif(not (MODELS_DIR / "meta.joblib").exists(), reason="models not trained")


@pytest.fixture(scope="module")
def az():
    from scam_analyzer.risk import ScamAnalyzer
    return ScamAnalyzer()


def _ok(r):
    assert r["ok"], r
    assert 0.0 <= r["score"] <= 1.0
    assert r["risk"] in {"LOW", "MEDIUM", "HIGH"}
    assert r["prediction"] in {"SCAM", "NON-SCAM"}
    assert r["explanation"] and r["recommendations"]
    assert r["stats"]["words"] >= 0


def test_01_clearly_legitimate(az):
    r = az.analyze(EXAMPLES["Legitimate appointment reminder"])
    _ok(r)
    assert r["prediction"] == "NON-SCAM" and r["risk"] == "LOW"


def test_02_obvious_scam(az):
    r = az.analyze(EXAMPLES["Lottery / prize with processing fee (scam)"])
    _ok(r)
    assert r["prediction"] == "SCAM" and r["risk"] == "HIGH"


def test_03_otp_request(az):
    r = az.analyze("Hello, I am from your bank. To complete verification please tell me the OTP sent to your phone right now.")
    _ok(r)
    assert r["prediction"] == "SCAM"
    assert "credential_request" in {d["key"] for d in r["indicators"]["indicators"]}


def test_04_bank_impersonation(az):
    r = az.analyze(EXAMPLES["Bank impersonation + OTP request (scam)"])
    _ok(r)
    assert r["prediction"] == "SCAM" and r["risk"] == "HIGH"
    assert "impersonation_authority" in {d["key"] for d in r["indicators"]["indicators"]}


def test_05_urgency_and_threats(az):
    r = az.analyze(EXAMPLES["Courier / police threat — 'digital arrest' (scam)"])
    _ok(r)
    assert r["prediction"] == "SCAM"
    assert "urgency_threat" in {d["key"] for d in r["indicators"]["indicators"]}


def test_06_ambiguous(az):
    r = az.analyze(EXAMPLES["Ambiguous — job offer"])
    _ok(r)   # no fixed expectation: the point is a graceful, explained answer
    assert r["indicators"]["indicators"], "the 'registration fee' should at least raise a rule flag"


def test_07_empty_input(az):
    for t in ("", "   ", None):
        r = az.analyze(t)
        assert r["ok"] is False and "paste" in r["error"].lower()


def test_08_very_short_input(az):
    r = az.analyze("hi")
    _ok(r)
    assert any("short" in w.lower() for w in r["warnings"])


def test_09_very_long_input(az):
    long_text = (EXAMPLES["Genuine bank fraud-alert call (legitimate)"] + "\n") * 600   # > 60k chars
    r = az.analyze(long_text)
    _ok(r)
    assert r["truncated"] is True


def test_10_irrelevant_text(az):
    r = az.analyze("def f(x): return x ** 2  # 12345 !!! ### λλλ ∑∑∑")
    assert (not r["ok"]) or r["warnings"], "irrelevant input must be rejected or come with a low-evidence warning"


def test_symbols_only_rejected(az):
    r = az.analyze("!!! ??? ### 🙂🙂")
    assert r["ok"] is False


@pytest.mark.parametrize("key", ["logistic_regression", "naive_bayes", "linear_svm_calibrated", "hard_voting",
                                 "soft_voting", "stacking", "bagging", "adaboost"])
def test_every_model_loads_and_predicts(az, key):
    r = az.analyze(EXAMPLES["Remote-access tech support (scam)"], model_key=key)
    _ok(r)


def test_genuine_bank_alert_rules_see_no_request(az):
    r = az.analyze(EXAMPLES["Genuine bank fraud-alert call (legitimate)"])
    _ok(r)
    assert "credential_request" not in {d["key"] for d in r["indicators"]["indicators"]}
    assert len(r["indicators"]["safe_signals"]) >= 2


@pytest.mark.xfail(reason="KNOWN LIMITATION (documented): in the training data almost every conversation about a bank card "
                          "is a scam, so a bag-of-words model scores a genuine bank fraud-alert call as high risk.", strict=False)
def test_genuine_bank_alert_model_not_high(az):
    r = az.analyze(EXAMPLES["Genuine bank fraud-alert call (legitimate)"])
    assert r["risk"] != "HIGH"
