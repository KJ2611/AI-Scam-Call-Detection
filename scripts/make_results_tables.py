"""Print Markdown result tables from reports/metrics.json (used to fill README.md)."""
import json
from pathlib import Path

m = json.loads((Path(__file__).resolve().parents[1] / "reports" / "metrics.json").read_text())
M, best = m["models"], m["best_model"]


def p(x, d=2):
    return "—" if x is None else f"{x*100:.{d}f}"


print("| Model | CV F1 (5-fold, train) | Test acc. | Precision | Recall | F1 | ROC-AUC | Brier | FN | FP | Ext. acc. | Ext. recall |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for k, r in sorted(M.items(), key=lambda kv: -kv[1]["test"]["main_test"]["f1"]):
    t, e = r["test"]["main_test"], r["test"]["ext_test"]
    cv = f"{p(r['cv']['f1']['mean'])} ± {p(r['cv']['f1']['std'])}" if r.get("cv") else "—"
    name = r["info"]["name"] + (" **(default)**" if k == best else "")
    brier = f"{t['brier']:.4f}" if t.get("brier") is not None else "—"
    auc = f"{t['roc_auc']:.4f}" if t.get("roc_auc") is not None else "—"
    print(f"| {name} | {cv} | {p(t['accuracy'])} | {p(t['precision'])} | {p(t['recall'])} | {p(t['f1'])} | {auc} | {brier} | "
          f"{t['false_negatives']} | {t['false_positives']} | {p(e['accuracy'], 1)} | {p(e['recall'], 1)} |")
print()
b = M[best]["test"]
print("| Test set (default model) | n | Accuracy | Precision | Recall | F1 |")
print("|---|---|---|---|---|---|")
for s, label in m["eval_sets"].items():
    t = b[s]
    print(f"| {label} | {t['n']:,} | {p(t['accuracy'])} | {p(t['precision'])} | {p(t['recall'])} | {p(t['f1'])} |")
print()
a = m["ablations"]
print("| Ablation | Result |")
print("|---|---|")
for k, v in a["historical_reproduction"]["models"].items():
    print(f"| Notebook {k} accuracy (reported → reproduced) | {p(v['reported_accuracy'])} → {p(v['reproduced_accuracy'])} |")
f = a["format_length_shortcut"]
print(f"| Format/length-only Random Forest (no words) | {p(f['format_length_only_rf_accuracy'])} accuracy (majority baseline {p(f['majority_baseline_accuracy'])}) |")
for k, v in a["out_of_domain_english"].items():
    print(f"| Out-of-domain: {M[k]['info']['name']} trained on composite only → all English scripts | {p(v['accuracy'])} acc., {p(v['recall'])} recall |")
mk = a["artifact_masking"]
print(f"| LR external accuracy, names masked vs kept | {p(mk['masked']['ext_test']['accuracy'])} vs {p(mk['unmasked']['ext_test']['accuracy'])} |")
nd = a["near_duplicates"]
print(f"| Test rows with a near-duplicate (cos ≥ 0.9) in training | {p(nd['share_of_main_test_with_train_neighbour'], 1)}%; default-model accuracy without them {p(b['main_test_no_near_duplicates']['accuracy'])} |")
