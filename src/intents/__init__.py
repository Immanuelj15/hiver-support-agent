"""
Intent classification package for Hiver AI Support Agent.
"""
from src.intents.taxonomy import Intent, ALL_INTENTS, SENSITIVE_INTENTS, INTENT_DESCRIPTIONS
from src.intents.confidence import calibrate_confidence
from src.intents.classifier import IntentClassifier

__all__ = [
    "Intent",
    "ALL_INTENTS",
    "SENSITIVE_INTENTS",
    "INTENT_DESCRIPTIONS",
    "calibrate_confidence",
    "IntentClassifier",
]
