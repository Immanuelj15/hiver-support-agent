"""
tests/test_escalation.py

Unit tests for deterministic escalation policies and safety rules.
"""

from src.escalation.policy import EscalationPolicy
from src.escalation.decision import EscalationAction, RiskLevel

def test_sensitive_intent_escalation():
    policy = EscalationPolicy()
    decision = policy.evaluate(
        customer_msg="I need to reset my password",
        predicted_intent="account_security_and_login",
        intent_confidence=0.95,
        retrieval_similarity=0.85
    )
    assert decision.action == EscalationAction.ESCALATE
    assert decision.should_escalate is True
    assert decision.triggered_rule == "SENSITIVE_INTENT_RULE"

def test_legal_threat_escalation():
    policy = EscalationPolicy()
    decision = policy.evaluate(
        customer_msg="I will sue your company and contact my lawyer!",
        predicted_intent="feedback_or_complaint",
        intent_confidence=0.95,
        retrieval_similarity=0.85
    )
    assert decision.action == EscalationAction.ESCALATE
    assert decision.triggered_rule == "LEGAL_REGULATORY_TRIGGER"
    assert decision.risk_level == RiskLevel.HIGH

def test_safety_hazard_escalation():
    policy = EscalationPolicy()
    decision = policy.evaluate(
        customer_msg="The battery on this device is smoking and exploded on my desk.",
        predicted_intent="damaged_or_wrong_item",
        intent_confidence=0.95,
        retrieval_similarity=0.85
    )
    assert decision.action == EscalationAction.ESCALATE
    assert decision.triggered_rule == "SAFETY_HAZARD_TRIGGER"
    assert decision.risk_level == RiskLevel.CRITICAL

def test_low_confidence_escalation():
    policy = EscalationPolicy(confidence_threshold=0.65)
    decision = policy.evaluate(
        customer_msg="Where is my package?",
        predicted_intent="order_delivery_delay",
        intent_confidence=0.45,  # Below 0.65
        retrieval_similarity=0.85
    )
    assert decision.action == EscalationAction.ESCALATE
    assert decision.triggered_rule == "LOW_CONFIDENCE_THRESHOLD"

def test_low_similarity_escalation():
    policy = EscalationPolicy(similarity_threshold=0.55)
    decision = policy.evaluate(
        customer_msg="Where is my package?",
        predicted_intent="order_delivery_delay",
        intent_confidence=0.95,
        retrieval_similarity=0.40  # Below 0.55
    )
    assert decision.action == EscalationAction.ESCALATE
    assert decision.triggered_rule == "LOW_RETRIEVAL_SIMILARITY"

def test_safe_auto_handle():
    policy = EscalationPolicy()
    decision = policy.evaluate(
        customer_msg="Where is my tracking number?",
        predicted_intent="order_delivery_delay",
        intent_confidence=0.95,
        retrieval_similarity=0.88
    )
    assert decision.action == EscalationAction.AUTO_HANDLE
    assert decision.should_escalate is False
    assert decision.risk_level == RiskLevel.LOW

def test_unverified_financial_promise():
    policy = EscalationPolicy()
    violation = policy.validate_reply("I have refunded your $50 to your card.")
    assert violation is not None
    assert "UNVERIFIED_FINANCIAL_PROMISE" in violation

    clean_reply = "You can view your tracking details in Your Orders here: [link]."
    assert policy.validate_reply(clean_reply) is None

def test_unsupported_question_no_evidence_escalation():
    """Unsupported question with zero or near-zero historical evidence must escalate."""
    policy = EscalationPolicy(similarity_threshold=0.55)
    decision = policy.evaluate(
        customer_msg="Can I use Amazon Prime on my Tesla in-car dashboard while driving across Mars?",
        predicted_intent="other",
        intent_confidence=0.50,
        retrieval_similarity=0.22  # Extremely low similarity
    )
    assert decision.action == EscalationAction.ESCALATE
    assert decision.should_escalate is True
    assert decision.triggered_rule in ("LOW_RETRIEVAL_SIMILARITY", "LOW_CONFIDENCE_THRESHOLD", "SENSITIVE_INTENT_RULE")


def test_unknown_issue_low_similarity_escalation():
    """Unknown issue with weak retrieval grounding must not auto-handle."""
    policy = EscalationPolicy(similarity_threshold=0.55)
    decision = policy.evaluate(
        customer_msg="The courier left a live badger in my recycling bin and it bit my neighbor.",
        predicted_intent="feedback_or_complaint",
        intent_confidence=0.60,
        retrieval_similarity=0.38  # Far below threshold
    )
    assert decision.action == EscalationAction.ESCALATE
    assert decision.should_escalate is True

