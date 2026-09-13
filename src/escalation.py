"""
src/escalation.py

Deterministic and rule-governed escalation logic:
Separates generation from routing decisions using an explicit multi-tiered rubric.
Outputs: {decision: "auto" | "escalate", reason: str}
"""

import os
import sys
import re
from typing import Dict, Any
from pathlib import Path

# Ensure repo root in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    DEFAULT_SIMILARITY_THRESHOLD,
    SENSITIVE_INTENTS
)

# High-risk crisis / legal / fraud keyword triggers
HIGH_RISK_KEYWORDS = [
    r"\blawyer\b", r"\battorney\b", r"\bsue\b", r"\blawsuit\b",
    r"\bcourt\b", r"\blegal action\b", r"\bpolice\b",
    r"\bchargeback\b", r"\bfraud\b", r"\bhacked\b", r"\bstolen identity\b",
    r"\bcredit card fraud\b", r"\bself[- ]harm\b", r"\bsuicide\b",
    r"\bbetter business bureau\b", r"\bbbb\b", r"\bftc\b"
]

# Action patterns that an AI bot cannot execute without live backend APIs
UNVERIFIABLE_ACTION_PATTERNS = [
    r"i have (issued|processed|applied|refunded|credited)\b",
    r"we have (issued|processed|applied|refunded|credited)\b",
    r"i have (canceled|cancelled|modified) your (order|subscription|account)\b",
    r"we have (canceled|cancelled|modified) your (order|subscription|account)\b",
    r"refund (of|for) \$?\d+",
    r"i (checked|looked up|verified) your account and\b",
    r"credited your (card|bank|account)\b",
    r"send (your|me) (password|pin|cvv|full credit card)\b"
]

class EscalationPolicy:
    """Evaluates whether an inquiry can be auto-handled or must be escalated to a human."""
    
    def __init__(self, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        self.similarity_threshold = similarity_threshold
        
    def evaluate(
        self,
        customer_msg: str,
        intent: str,
        confidence: float,
        max_similarity: float,
        reply_draft: str
    ) -> Dict[str, str]:
        """
        Applies hierarchical rules in deterministic priority order.
        First match determines the decision.
        """
        msg_lower = customer_msg.lower()
        reply_lower = reply_draft.lower()
        
        # Rule 1: High-risk or ambiguous intents
        if intent in SENSITIVE_INTENTS:
            return {
                "decision": "escalate",
                "reason": f"sensitive/ambiguous intent: {intent}"
            }
            
        # Rule 1b: Emergency / Legal / Fraud keyword triggers
        for pattern in HIGH_RISK_KEYWORDS:
            if re.search(pattern, msg_lower):
                return {
                    "decision": "escalate",
                    "reason": f"detected high-risk trigger: '{pattern}' in customer message"
                }
                
        # Rule 2: Low retrieval confidence (no strong historical precedent)
        if max_similarity < self.similarity_threshold:
            return {
                "decision": "escalate",
                "reason": (
                    f"no closely similar historical resolution found "
                    f"(max_sim={max_similarity:.3f} < threshold={self.similarity_threshold:.2f})"
                )
            }
            
        # Rule 3: Low classifier confidence
        if confidence < 0.65:
            return {
                "decision": "escalate",
                "reason": f"low intent classification confidence ({confidence:.2f} < 0.65)"
            }
            
        # Rule 4: Reply draft implies unverifiable actions (financial promises, backend DB writes)
        for act_pattern in UNVERIFIABLE_ACTION_PATTERNS:
            if re.search(act_pattern, reply_lower):
                return {
                    "decision": "escalate",
                    "reason": (
                        "requires account verification/action the agent cannot perform "
                        f"(triggered by draft pattern: '{act_pattern}')"
                    )
                }
                
        # Rule 5: Safe auto-resolution criteria met
        return {
            "decision": "auto",
            "reason": f"similar historical case resolved with same approach (sim={max_similarity:.3f}, intent={intent})"
        }

def escalate_or_handle(
    message: str,
    intent: str,
    confidence: float,
    max_similarity: float,
    reply_draft: str,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD
) -> Dict[str, str]:
    """Convenience helper function."""
    policy = EscalationPolicy(similarity_threshold=threshold)
    return policy.evaluate(
        customer_msg=message,
        intent=intent,
        confidence=confidence,
        max_similarity=max_similarity,
        reply_draft=reply_draft
    )

if __name__ == "__main__":
    policy = EscalationPolicy()
    
    # Test case 1: Security inquiry
    res1 = policy.evaluate(
        customer_msg="My account was hacked and someone changed my email",
        intent="account_security_and_login",
        confidence=0.98,
        max_similarity=0.85,
        reply_draft="Please send us a DM to secure your account."
    )
    print("Test 1 (Security):", res1)
    
    # Test case 2: Low similarity inquiry
    res2 = policy.evaluate(
        customer_msg="Can my dog eat the cardboard box your courier delivered?",
        intent="damaged_or_wrong_item",
        confidence=0.70,
        max_similarity=0.38,
        reply_draft="We advise keeping packing materials away from pets."
    )
    print("Test 2 (Low Sim):", res2)
    
    # Test case 3: Unverifiable action promise
    res3 = policy.evaluate(
        customer_msg="My package is late, give me my money back right now.",
        intent="order_delivery_delay",
        confidence=0.95,
        max_similarity=0.78,
        reply_draft="I have issued a refund of $45 to your credit card."
    )
    print("Test 3 (Unverifiable Action):", res3)
    
    # Test case 4: Safe auto-response
    res4 = policy.evaluate(
        customer_msg="How can I track my package? The tracking page is giving an error.",
        intent="order_delivery_delay",
        confidence=0.96,
        max_similarity=0.74,
        reply_draft="You can view live shipment updates by navigating to Your Orders on our website or mobile app."
    )
    print("Test 4 (Safe Auto):", res4)
