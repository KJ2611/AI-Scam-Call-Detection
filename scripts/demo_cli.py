"""Terminal fallback for the demo:  python scripts/demo_cli.py "transcript text"   (or pipe text via stdin)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scam_analyzer.risk import ScamAnalyzer  # noqa: E402


def main():
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else sys.stdin.read()
    r = ScamAnalyzer().analyze(text)
    if not r["ok"]:
        print("Error:", r["error"])
        return
    print(f"\nPrediction : {r['prediction']}   Risk: {r['risk']}   P(scam) = {r['score']:.1%}   model: {r['model_name']}")
    if r["escalation"]:
        print("Escalation :", r["escalation"])
    for w in r["warnings"]:
        print("Warning    :", w)
    print("\nToward scam     :", ", ".join(f"{t} ({c:+.2f})" for t, c, *_ in r["evidence"]["toward_scam"][:8]) or "—")
    print("Toward legit    :", ", ".join(f"{t} ({c:+.2f})" for t, c, *_ in r["evidence"]["toward_non_scam"][:8]) or "—")
    print("\nRed flags:")
    for d in r["indicators"]["indicators"]:
        print(f"  [{d['severity']}] {d['label']}: {', '.join(d['phrases'][:5])}")
    if r["indicators"]["safe_signals"]:
        print("Safe signals:", ", ".join(r["indicators"]["safe_signals"]))
    print("\nAdvice:")
    for t in r["recommendations"][:5]:
        print("  -", t)
    print("\n" + r["disclaimer"])


if __name__ == "__main__":
    main()
