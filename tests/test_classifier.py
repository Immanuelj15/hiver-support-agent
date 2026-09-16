"""
tests/test_classifier.py

Unit tests for intent classifier schema, parsing, and confidence calibration.
"""

import pytest
from src.intents.taxonomy import Intent, ALL_INTENTS, SENSITIVE_INTENTS
from src.intents.confidence import calibrate_confidence
from src.intents.classifier import IntentClassifier

class MockLLMProvider:
    def __init__(self, return_text: str):
        self.return_text = return_text
        self.model_name = "mock-model"
        self.provider_type = "mock"

    def generate(self, prompt, **kwargs):
        return self.return_text

def test_intent_taxonomy():
    assert len(ALL_INTENTS) == 8
    assert "order_delivery_delay" in ALL_INTENTS
    assert "account_security_and_login" in ALL_INTENTS
    assert "account_security_and_login" in SENSITIVE_INTENTS

def test_confidence_calibration():
    # Regular score
    score = calibrate_confidence(0.95, "order_delivery_delay")
    assert 0.90 <= score <= 0.99

    # Score clamping
    low_score = calibrate_confidence(-0.5, "order_delivery_delay")
    assert low_score >= 0.10

    high_score = calibrate_confidence(1.5, "order_delivery_delay")
    assert high_score <= 0.99

    # Ambiguity penalty on 'other'
    other_score = calibrate_confidence(0.90, "other")
    assert other_score < 0.90

def test_classifier_json_parsing():
    json_resp = '```json\n{"intent": "order_delivery_delay", "confidence": 0.95, "reasoning": "Late package inquiry."}\n```'
    mock_llm = MockLLMProvider(json_resp)
    classifier = IntentClassifier(llm_provider=mock_llm)

    res = classifier.classify("Where is my parcel?")
    assert res["intent"] == "order_delivery_delay"
    assert res["confidence"] >= 0.90
    assert "Late package" in res["reasoning"]

def test_classifier_empty_input():
    mock_llm = MockLLMProvider("{}")
    classifier = IntentClassifier(llm_provider=mock_llm)

    res = classifier.classify("   ")
    assert res["intent"] == Intent.OTHER.value
    assert res["confidence"] <= 0.30

def test_classifier_fallback_heuristic():
    mock_llm = MockLLMProvider("MALFORMED NON-JSON TEXT")
    classifier = IntentClassifier(llm_provider=mock_llm)

    res = classifier.classify("Someone hacked my password and account!")
    assert res["intent"] == "account_security_and_login"
