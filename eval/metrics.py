"""
eval/metrics.py

Automated Evaluation Metrics:
- Intent Classification: Accuracy, Macro-F1, Weighted-F1, Confusion Matrix
- Escalation Routing: Precision, Recall, F1, Specificity (treating 'escalate' as positive class)
- Retrieval Quality: Top-1 Retrieval Hit Rate (retrieved intent matches gold intent)
- Performance: Latency percentiles (P50, P95) and token cost estimation
"""

import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from src.config import INTENTS

def compute_intent_metrics(y_true: List[str], y_pred: List[str], labels: List[str] = INTENTS) -> Dict[str, Any]:
    """Computes accuracy, macro-F1, and per-class metrics for intent classification."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    
    # Per-intent metrics over all predefined intents
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    per_intent = {}
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        per_intent[label] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int(cm[i, :].sum())
        }
        
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_intent": per_intent,
        "confusion_matrix_labels": labels,
        "confusion_matrix": cm.tolist()
    }

def compute_escalation_metrics(y_true_bool: List[bool], y_pred_decision: List[str]) -> Dict[str, Any]:
    """
    Computes Precision, Recall, F1, and False Negative Rate for escalation routing.
    'escalate' is the POSITIVE class.
    """
    # Convert string decisions to boolean: 'escalate' -> True, 'auto' -> False
    y_pred_bool = [d.lower() == "escalate" for d in y_pred_decision]
    
    acc = accuracy_score(y_true_bool, y_pred_bool)
    prec = precision_score(y_true_bool, y_pred_bool, zero_division=0)
    rec = recall_score(y_true_bool, y_pred_bool, zero_division=0)
    f1 = f1_score(y_true_bool, y_pred_bool, zero_division=0)
    
    # Calculate False Negatives (critical safety failures)
    # y_true=True (should escalate) but y_pred=False (auto-handled)
    false_negatives = sum(1 for yt, yp in zip(y_true_bool, y_pred_bool) if yt and not yp)
    false_positives = sum(1 for yt, yp in zip(y_true_bool, y_pred_bool) if not yt and yp)
    true_positives = sum(1 for yt, yp in zip(y_true_bool, y_pred_bool) if yt and yp)
    true_negatives = sum(1 for yt, yp in zip(y_true_bool, y_pred_bool) if not yt and not yp)
    
    fnr = false_negatives / max(sum(y_true_bool), 1)
    
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "false_negative_rate": round(fnr, 4),
        "escalation_rate": round(sum(y_pred_bool) / max(len(y_pred_bool), 1), 4)
    }

def compute_retrieval_hit_rate(gold_intents: List[str], retrieved_pairs_list: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Calculates top-1 retrieval hit rate: % of cases where the top-1 retrieved case
    belongs to the same intent as the gold label.
    """
    total = len(gold_intents)
    if total == 0:
        return {"top_1_hit_rate": 0.0, "total_queries": 0}
        
    top_1_hits = 0
    top_3_hits = 0
    
    for gold_intent, pairs in zip(gold_intents, retrieved_pairs_list):
        if not pairs:
            continue
        # Top-1
        top_1_intent = pairs[0].get("intent", "")
        if top_1_intent == gold_intent:
            top_1_hits += 1
            
        # Top-3
        intents_in_top_3 = [p.get("intent", "") for p in pairs[:3]]
        if gold_intent in intents_in_top_3:
            top_3_hits += 1
            
    return {
        "top_1_hit_rate": round(top_1_hits / total, 4),
        "top_3_hit_rate": round(top_3_hits / total, 4),
        "total_queries": total
    }

def compute_latency_and_cost(latencies_ms: List[float], total_tokens_est: int) -> Dict[str, Any]:
    """Computes latency percentiles and estimated Gemini/OpenAI cost."""
    if not latencies_ms:
        return {}
    arr = np.array(latencies_ms)
    # Gemini 2.5/3.6 Flash approx cost: $0.15 / 1M input tokens, $0.60 / 1M output tokens (~$0.00035 / 1k tokens)
    cost_est = (total_tokens_est / 1000.0) * 0.00035
    
    return {
        "mean_latency_ms": round(float(np.mean(arr)), 1),
        "p50_latency_ms": round(float(np.median(arr)), 1),
        "p90_latency_ms": round(float(np.percentile(arr, 90)), 1),
        "p95_latency_ms": round(float(np.percentile(arr, 95)), 1),
        "estimated_total_cost_usd": round(cost_est, 5),
        "estimated_cost_per_inquiry_usd": round(cost_est / max(len(latencies_ms), 1), 6)
    }
