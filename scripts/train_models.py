"""Entry point:  python scripts/train_models.py [--fast] [--folds 5]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scam_analyzer.train import main  # noqa: E402

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Train and evaluate all scam-detection models")
    ap.add_argument("--folds", type=int, default=5, help="CV folds for model selection (default 5)")
    ap.add_argument("--fast", action="store_true", help="skip CV for the slow Bagging/AdaBoost models")
    a = ap.parse_args()
    main(a.folds, a.fast)
