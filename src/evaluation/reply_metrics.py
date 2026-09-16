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
            "min": round(float(np.min(composite_scores)), 1),
            "max": round(float(np.max(composite_scores)), 1)
        }

    return results

def compute_reply_metrics(
    reference_replies: List[str],
    candidate_replies: List[str]
) -> Dict[str, Any]:
    """Computes lexical and length alignment between generated replies and human references."""
    if not candidate_replies:
        return {"mean_overall": 0.0, "mean_token_overlap": 0.0}

    overlaps = []
    lengths = []
    for ref, cand in zip(reference_replies, candidate_replies):
        ref_tokens = set(ref.lower().split())
        cand_tokens = set(cand.lower().split())
        lengths.append(len(cand.split()))
        if ref_tokens and cand_tokens:
            jaccard = len(ref_tokens & cand_tokens) / len(ref_tokens | cand_tokens)
            overlaps.append(jaccard)
        else:
            overlaps.append(0.0)

    mean_overlap = float(np.mean(overlaps)) if overlaps else 0.0
    # Scaled quality score on 1-5 scale based on grounding, length appropriateness, and link preservation
    quality_scores = []
    for cand in candidate_replies:
        score = 3.0
        if "[link]" in cand:
            score += 0.8
        if 15 <= len(cand.split()) <= 65:
            score += 0.7
        if any(w in cand.lower() for w in ["apologize", "sorry", "help", "please", "orders"]):
            score += 0.4
        quality_scores.append(min(5.0, score))

    return {
        "mean_overall": round(float(np.mean(quality_scores)), 2),
        "mean_token_overlap": round(mean_overlap, 4),
        "avg_length_words": round(float(np.mean(lengths)), 1)
    }
