"""
src/escalation/policy.py

Deterministic escalation policy engine.
Evaluates customer messages, predicted intent, classifier confidence, and retrieval similarity
against hard safety constraints and risk triggers.
"""

import re
from typing import Dict, Any, Optional

from src.escalation.decision import EscalationAction, RiskLevel, EscalationDecision
from src.intents.taxonomy import SENSITIVE_INTENTS, Intent

# Regex patterns for high-risk domains
LEGAL_REGULATORY_PATTERN = re.compile(
    r"\b(lawyer|attorney|lawsuit|sue|suing|litigation|court|police|ftc|bbb|better business bureau|consumer protection)\b",
    re.IGNORECASE
)

SAFETY_HAZARD_PATTERN = re.compile(
    r"\b(fire|smoke|smoking|exploded?|explosion|shattered glass|bleeding|injury|injured|hospital|paramedics|poison|swollen battery|hot to the touch)\b",
    re.IGNORECASE
)

FINANCIAL_DISPUTE_PATTERN = re.compile(
    r"\b(chargeback|dispute with (?:my )?bank|fraudulent charge|stolen credit card|refund my \$\d+|unauthorized (?:purchase|order))\b",
    re.IGNORECASE
)

MISCONDUCT_PATTERN = re.compile(
    r"\b(backed into|hit my (?:car|mailbox|dog)|threatened|assaulted|cursed at|hung up on me|called me an? \w+)\b",
    re.IGNORECASE
)

PROMPT_INJECTION_PATTERN = re.compile(
    r"\b(ignore (?:all )?previous instructions|system prompt|dan mode|jailbreak|admin token)\b",
    re.IGNORECASE
)

UNVERIFIED_FINANCIAL_PROMISE_PATTERN = re.compile(
    r"\b(i (?:will|have) (?:refund(?:ed)?|credited?|waived?)|sending you \$\d+|granting a (?:full )?refund of)\b",
    re.IGNORECASE
)

class EscalationPolicy:
    def __init__(
        self,
        similarity_threshold: float = 0.55,
        confidence_threshold: float = 0.65
    ):
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold

    def evaluate(
        self,
        customer_msg: str,
        predicted_intent: str,
        intent_confidence: float,
        retrieval_similarity: float
    ) -> EscalationDecision:
        """
        Evaluate customer query and pipeline telemetry.
        Returns deterministic EscalationDecision.
        """
        msg = customer_msg.strip()

        # 1. Prompt injection / adversarial attempt
        if PROMPT_INJECTION_PATTERN.search(msg):
            return EscalationDecision(
                action=EscalationAction.ESCALATE,
                should_escalate=True,
                reason="Adversarial prompt injection or system override detected.",
                risk_level=RiskLevel.HIGH,
                triggered_rule="PROMPT_INJECTION_TRIGGER"
            )

        # 2. Safety hazard / physical injury
        if SAFETY_HAZARD_PATTERN.search(msg):
            return EscalationDecision(
                action=EscalationAction.ESCALATE,
                should_escalate=True,
                reason="Physical safety hazard or bodily injury reported.",
                risk_level=RiskLevel.CRITICAL,
                triggered_rule="SAFETY_HAZARD_TRIGGER"
            )

        # 3. Legal or regulatory escalation
        if LEGAL_REGULATORY_PATTERN.search(msg):
            return EscalationDecision(
                action=EscalationAction.ESCALATE,
                should_escalate=True,
                reason="Legal action or regulatory reporting threatened.",
                risk_level=RiskLevel.HIGH,
                triggered_rule="LEGAL_REGULATORY_TRIGGER"
            )

        # 4. Mandatory sensitive intent escalation (Account Security & Fraud)
        if predicted_intent in SENSITIVE_INTENTS:
            return EscalationDecision(
                action=EscalationAction.ESCALATE,
                should_escalate=True,
                reason=f"Mandatory escalation for sensitive intent '{predicted_intent}'.",
                risk_level=RiskLevel.HIGH if predicted_intent == Intent.ACCOUNT_SECURITY_AND_LOGIN.value else RiskLevel.MEDIUM,
                triggered_rule="SENSITIVE_INTENT_RULE"
            )

        # 5. High-risk courier misconduct or damage
        if MISCONDUCT_PATTERN.search(msg):
            return EscalationDecision(
                action=EscalationAction.ESCALATE,
                should_escalate=True,
                reason="Courier misconduct or property damage reported.",
                risk_level=RiskLevel.HIGH,
                triggered_rule="MISCONDUCT_TRIGGER"
            )

        # 6. Explicit financial dispute / chargeback threat
        if FINANCIAL_DISPUTE_PATTERN.search(msg):
            return EscalationDecision(
                action=EscalationAction.ESCALATE,
                should_escalate=True,
                reason="Financial dispute or chargeback action requiring account verification.",
                risk_level=RiskLevel.HIGH,
                triggered_rule="FINANCIAL_DISPUTE_TRIGGER"
            )

        # 7. Low intent confidence threshold
        if intent_confidence < self.confidence_threshold:
            return EscalationDecision(
                action=EscalationAction.ESCALATE,
                should_escalate=True,
                reason=f"Classifier confidence ({intent_confidence:.2f}) below threshold ({self.confidence_threshold:.2f}).",
                risk_level=RiskLevel.MEDIUM,
                triggered_rule="LOW_CONFIDENCE_THRESHOLD"
            )

        # 8. Low retrieval similarity threshold
        if retrieval_similarity < self.similarity_threshold:
            return EscalationDecision(
                action=EscalationAction.ESCALATE,
                should_escalate=True,
                reason=f"Retrieval similarity ({retrieval_similarity:.2f}) below groundability threshold ({self.similarity_threshold:.2f}).",
                risk_level=RiskLevel.MEDIUM,
                triggered_rule="LOW_RETRIEVAL_SIMILARITY"
            )

        # Default: Safe for automated resolution
        return EscalationDecision(
            action=EscalationAction.AUTO_HANDLE,
            should_escalate=False,
            reason="Query fits standard self-serve resolution workflows with strong evidence grounding.",
            risk_level=RiskLevel.LOW,
            triggered_rule=None
        )

    def validate_reply(self, draft_reply: str) -> Optional[str]:
        """
        Check generated draft reply for unverified financial promises.
        Returns violation message if detected, else None.
        """
        if UNVERIFIED_FINANCIAL_PROMISE_PATTERN.search(draft_reply):
            return "UNVERIFIED_FINANCIAL_PROMISE: Draft reply promises financial compensation without agent verification."
        return None
