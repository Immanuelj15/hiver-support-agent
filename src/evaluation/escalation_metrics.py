"""
src/evaluation/escalation_metrics.py

Safety and operational metrics for deterministic escalation policy.
Measures precision, recall, false escalation rate, and missed escalation rate.
"""

from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def compute_escalation_metrics(
    y_true_escalate: List[bool],
    y_pred_escalate: List[bool]
) -> Dict[str, Any]:
    """
    Computes safety-critical escalation metrics:
    - Missed Escalations (FN): Safety violations / angry customers auto-handled erroneously.
    - False Escalations (FP): Benign queries unnecessarily sent to human queues.
    """
    y_true = [bool(x) for x in y_true_escalate]
    y_pred = [bool(x) for x in y_pred_escalate]

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()

    missed_rate = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    false_alarm_rate = fp / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "sample_count": len(y_true),
        "accuracy": round(float(acc), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1_score": round(float(f1), 4),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "missed_escalation_rate": round(float(missed_rate), 4),
        "false_auto_handle_rate": round(float(missed_rate), 4),
        "false_escalation_rate": round(float(false_alarm_rate), 4)
    }

