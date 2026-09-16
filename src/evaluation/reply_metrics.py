"""
src/evaluation/reply_metrics.py

Defines the 6 response quality dimensions and computes aggregate score distributions.
Dimensions: Groundedness, Correctness, Relevance, Helpfulness, Tone, Resolution.
"""

from typing import List, Dict, Any
import numpy as np

RUBRIC_CRITERIA = {
    "groundedness": "Faithfulness to retrieved evidence; zero hallucinated order details or false policy claims.",
    "correctness": "Factual accuracy under official Amazon customer support protocols.",
    "relevance": "Directly addresses the customer's specific grievance or question without generic filler.",
    "helpfulness": "Provides clear, actionable guidance or links (e.g. [link], return QR code, safe locations).",
    "tone": "Empathetic, polite, professional, and de-escalating in tone.",
    "resolution": "Provides a definitive closure, self-serve step, or clear escalation handoff."
}

def aggregate_reply_scores(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes mean, median, min, max, and pass rates for the 6 quality dimensions."""
    if not evaluations:
        return {}

    dimensions = list(RUBRIC_CRITERIA.keys())
    results = {}
    composite_scores = []

    for dim in dimensions:
        scores = [float(e.get(dim, 3.0)) for e in evaluations if dim in e]
        if scores:
            results[dim] = {
                "mean": round(float(np.mean(scores)), 2),
                "median": round(float(np.median(scores)), 2),
                "std": round(float(np.std(scores)), 2),
                "min": round(float(np.min(scores)), 1),
                "max": round(float(np.max(scores)), 1),
                "pass_rate_ge_4": round(float(np.mean([s >= 4.0 for s in scores])), 4)
            }

    for e in evaluations:
        row_scores = [float(e.get(dim, 3.0)) for dim in dimensions if dim in e]
        if row_scores:
            composite_scores.append(np.mean(row_scores))

    if composite_scores:
        results["composite_quality"] = {
            "mean": round(float(np.mean(composite_scores)), 2),
            "median": round(float(np.median(composite_scores)), 2),
            "std": round(float(np.std(composite_scores)), 2),
            "overall_pass_rate_ge_3_5": round(float(np.mean([s >= 3.5 for s in composite_scores])), 4)
        }

    return results
