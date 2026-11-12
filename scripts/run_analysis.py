"""Build the EDA and advanced (clustering + HMM) reports used by the app.

Run after training:   python scripts/run_analysis.py      (≈ 3–5 minutes)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scam_analyzer import advanced, eda  # noqa: E402

if __name__ == "__main__":
    print("EDA …", flush=True)
    eda.main()
    print("Clustering + HMM …", flush=True)
    advanced.main()
    print("Done: reports/eda.json, reports/advanced.json, models/hmm_models.joblib")
