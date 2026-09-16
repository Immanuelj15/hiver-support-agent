"""
src/evaluation/intent_metrics.py

Evaluation metrics for intent classification.
Calculates Accuracy, Macro/Weighted Precision, Recall, F1, and per-class breakdowns.
"""

from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

def compute_intent_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Compute standard intent classification performance metrics."""
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    labels = sorted(list(set(y_true) | set(y_pred)))
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    per_class = {}
    for label in labels:
        if label in report:
            per_class[label] = {
                "precision": round(report[label]["precision"], 4),
                "recall": round(report[label]["recall"], 4),
                "f1": round(report[label]["f1-score"], 4),
                "support": report[label]["support"]
            }

    return {
        "sample_count": len(y_true),
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_w), 4),
        "per_class": per_class
    }
