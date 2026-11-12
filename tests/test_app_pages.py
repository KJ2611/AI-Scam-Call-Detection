"""Headless UI tests with Streamlit's AppTest: every page renders without an exception,
and the detection page produces a verdict for scam, legitimate and invalid input."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from scam_analyzer.config import MODELS_DIR, REPORTS_DIR

HARNESS = str(Path(__file__).with_name("app_harness.py"))
pytestmark = pytest.mark.skipif(not (MODELS_DIR / "meta.joblib").exists() or not (REPORTS_DIR / "advanced.json").exists(),
                                reason="models / reports not built")


def _run(page):
    at = AppTest.from_file(HARNESS, default_timeout=120)
    at.query_params["page"] = page
    at.run()
    return at


@pytest.mark.parametrize("page", ["home", "detect", "models", "dataset", "advanced", "about"])
def test_page_renders(page):
    at = _run(page)
    assert not at.exception, [e.message for e in at.exception]
    assert at.title, "page should have a title"


def _analyze(text):
    at = _run("detect")
    at.text_area(key="transcript").set_value(text).run()
    next(b for b in at.button if b.label == "Analyze conversation").click().run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def test_detect_scam_flow():
    from examples import EXAMPLES  # noqa: F401  (path set by harness import above)
    at = _analyze("Sir this is your bank. Your account is blocked. Share the OTP immediately and pay the processing fee by UPI.")
    html = " ".join(m.value for m in at.markdown)
    assert "SCAM" in html and "Risk level" in html


def test_detect_legit_flow():
    at = _analyze("Hi mom, I reached the hostel safely. The train was on time. I'll call you after dinner, love you.")
    html = " ".join(m.value for m in at.markdown)
    assert "NON-SCAM" in html


def test_detect_empty_input_shows_error():
    at = _analyze("")
    assert at.error and "paste" in at.error[0].value.lower()
