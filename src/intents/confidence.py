"""
src/intents/confidence.py

Confidence estimation and score calibration for intent predictions.
Normalizes raw LLM likelihood or margin scores into calibrated probabilities [0.0, 1.0].
"""

import math
from typing import Dict, Any, Optional

def calibrate_confidence(
    raw_score: float,
    intent: str,
    ambiguity_penalty: float = 0.0,
    min_confidence: float = 0.1,
    max_confidence: float = 0.99
) -> float:
    """
    Calibrate a raw confidence score (e.g. from LLM self-assessment or logit margin)
    applying penalties for linguistic ambiguity and intent sensitivity.
    """
    score = max(0.0, min(1.0, float(raw_score)))
    score = score - ambiguity_penalty

    # Sensitive intents like 'other' inherently have higher variance, damp slightly
    if intent == "other":
        score *= 0.85

    score = max(min_confidence, min(max_confidence, score))
    return round(score, 4)
