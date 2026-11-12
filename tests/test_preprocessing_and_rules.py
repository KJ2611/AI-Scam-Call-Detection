"""Unit tests that need no trained model."""
import numpy as np
import pytest
from hmmlearn.hmm import CategoricalHMM

from scam_analyzer.hmm_algorithms import backward, forward, posteriors, unscaled_forward_probability, viterbi
from scam_analyzer.indicators import HMM_SYMBOLS, detect_indicators, text_to_symbols
from scam_analyzer.preprocessing import clean_text, legacy_clean_text, split_utterances


# ------------------------------------------------------------------ preprocessing
@pytest.mark.parametrize("raw,expected_tokens", [
    ("Call +91 98765 43210 now", ["__phone__"]),
    ("visit http://bad-site.xyz/login", ["__url__"]),
    ("mail me at a.b@example.com", ["__email__"]),
    ("pay Rs. 5,000 today", ["__money__"]),
    ("you won 5 million rupees", ["__num__", "__money__", "won"]),
    ("send via PhonePe", ["__payapp__"]),
])
def test_placeholders(raw, expected_tokens):
    toks = clean_text(raw).split()
    for t in expected_tokens:
        assert t in toks, (raw, toks)


def test_empty_and_none():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text("   \n\t ") == ""
    assert clean_text("!!! ??? ...") == ""


def test_lowercase_and_punctuation():
    assert clean_text("HELLO, World!!") == "hello world"


def test_source_artifact_names_masked():
    assert "zhang" not in clean_text("I am Xiao Zhang from customer service").split()
    assert "zhang" in clean_text("I am Xiao Zhang", mask_source_artifacts=False).split()


def test_won_is_not_a_currency():
    assert "won" in clean_text("You have won a prize").split()


def test_legacy_cleaning_matches_notebook_style():
    assert legacy_clean_text("Prize of 5 million!") == "prize of <number> million"


def test_split_utterances_list_format():
    assert split_utterances("'Hello there', 'How are you?'") == ["Hello there", "How are you?"]


# ------------------------------------------------------------------ rule indicators
def test_otp_request_flagged_high():
    r = detect_indicators("Please read the OTP you just received to me immediately.")
    keys = {d["key"] for d in r["indicators"]}
    assert "credential_request" in keys and "urgency_threat" in keys
    assert r["n_high"] >= 1


def test_safe_warning_not_flagged_as_request():
    r = detect_indicators("The bank will never ask for your OTP, PIN or password. Never share your PIN with anyone.")
    assert "credential_request" not in {d["key"] for d in r["indicators"]}
    assert r["safe_signals"]


def test_remote_access_and_payment():
    r = detect_indicators("Download AnyDesk and pay the processing fee with a gift card.")
    keys = {d["key"] for d in r["indicators"]}
    assert {"remote_access", "payment_request"} <= keys


def test_benign_text_has_no_high_flags():
    r = detect_indicators("Are we still on for dinner tonight? Bring the book you borrowed.")
    assert r["n_high"] == 0


def test_hmm_symbols_are_valid():
    seq = text_to_symbols(clean_text("I am calling from the bank. Share your OTP now or your account will be blocked."), window=6)
    assert seq and all(0 <= s < len(HMM_SYMBOLS) for s in seq)
    assert HMM_SYMBOLS.index("CREDENTIAL") in seq


# ------------------------------------------------------------------ HMM algorithms vs hmmlearn
@pytest.fixture(scope="module")
def tiny_hmm():
    rng = np.random.RandomState(0)
    X = rng.randint(0, 4, size=(300, 1))
    m = CategoricalHMM(n_components=3, n_iter=20, random_state=0, n_features=4).fit(X, [30] * 10)
    return m


def test_forward_matches_hmmlearn(tiny_hmm):
    obs = np.array([0, 1, 3, 2, 2, 1, 0, 3, 3, 1])
    _, _, ll = forward(obs, tiny_hmm.startprob_, tiny_hmm.transmat_, tiny_hmm.emissionprob_)
    assert ll == pytest.approx(tiny_hmm.score(obs.reshape(-1, 1)), abs=1e-8)


def test_viterbi_matches_hmmlearn(tiny_hmm):
    obs = np.array([0, 1, 3, 2, 2, 1, 0, 3, 3, 1, 2, 0])
    path, lp = viterbi(obs, tiny_hmm.startprob_, tiny_hmm.transmat_, tiny_hmm.emissionprob_)
    ref_lp, ref_path = tiny_hmm.decode(obs.reshape(-1, 1), algorithm="viterbi")
    assert np.array_equal(path, ref_path)
    assert lp == pytest.approx(ref_lp, abs=1e-8)


def test_posteriors_sum_to_one_and_match(tiny_hmm):
    obs = np.array([3, 3, 0, 1, 2, 2, 1])
    a, c, _ = forward(obs, tiny_hmm.startprob_, tiny_hmm.transmat_, tiny_hmm.emissionprob_)
    g = posteriors(a, backward(obs, tiny_hmm.transmat_, tiny_hmm.emissionprob_, c))
    assert np.allclose(g.sum(axis=1), 1)
    assert np.allclose(g, tiny_hmm.predict_proba(obs.reshape(-1, 1)), atol=1e-8)


def test_scaling_prevents_underflow(tiny_hmm):
    obs = np.tile([0, 1, 2, 3], 300)  # T = 1200
    assert unscaled_forward_probability(obs, tiny_hmm.startprob_, tiny_hmm.transmat_, tiny_hmm.emissionprob_) == 0.0
    _, _, ll = forward(obs, tiny_hmm.startprob_, tiny_hmm.transmat_, tiny_hmm.emissionprob_)
    assert np.isfinite(ll) and ll < 0
