"""
baselines/majority.py

Baseline 1: Majority Class Classifier.
Always predicts the most frequent class in Amazon customer support: 'order_delivery_delay'.
Provides the empirical floor for classification performance.
"""

import sys
import json
import argparse
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.intents.taxonomy import Intent, ALL_INTENTS

def evaluate_majority(golden_path: str = "data/golden/golden_set.jsonl") -> dict:
    """Evaluate majority class baseline against golden evaluation set."""
    path = Path(golden_path)
    if not path.exists():
        # Fallback to legacy path if not yet generated
        path = Path("golden_set/golden_150.jsonl")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    y_true = [r["gold_intent"] for r in records]
    majority_class = Intent.ORDER_DELIVERY_DELAY.value
    y_pred = [majority_class] * len(y_true)

    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    results = {
        "baseline_name": "Majority Class ('order_delivery_delay')",
        "test_count": len(y_true),
        "majority_class": majority_class,
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_w), 4)
    }

    print("\n" + "=" * 60)
    print("BASELINE 1: MAJORITY CLASS EVALUATION")
    print("=" * 60)
    print(f"Evaluated on:       {path} (N={len(y_true)})")
    print(f"Predicted class:    {majority_class}")
    print(f"Accuracy:           {results['accuracy'] * 100:.2f}%")
    print(f"Macro Precision:    {results['macro_precision']:.4f}")
    print(f"Macro Recall:       {results['macro_recall']:.4f}")
    print(f"Macro F1 Score:     {results['macro_f1']:.4f}")
    print(f"Weighted F1:        {results['weighted_f1']:.4f}")
    print("=" * 60)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="data/golden/golden_set.jsonl", help="Path to golden set")
    args = parser.parse_args()
    evaluate_majority(args.golden)
