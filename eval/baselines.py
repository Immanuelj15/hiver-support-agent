"""
eval/baselines.py

Baseline Model Implementations:
1. Trivial Baseline:
   - Predicts majority intent (order_delivery_delay).
   - Always escalates (decision: 'escalate', reason: 'Trivial policy: 100% human escalation').
   - Empty/canned reply.

2. Simple Baseline:
   - TF-IDF + Logistic Regression intent classifier.
   - Fixed canned template reply: "Thanks for reaching out! Please send us a direct message with your order details so we can investigate."
   - Fixed naive escalation rule: escalate if and only if predicted intent is 'other'.
"""

import sys
from typing import Dict, Any, List
from pathlib import Path

# Ensure repo root in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.intent_classifier import BaselineTfidfClassifier

class TrivialBaseline:
    """Trivial Baseline: always predicts majority class and always escalates."""
    
    def __init__(self, majority_intent: str = "order_delivery_delay"):
        self.majority_intent = majority_intent
        
    def run(self, customer_msg: str) -> Dict[str, Any]:
        return {
            "customer_msg": customer_msg,
            "intent": self.majority_intent,
            "confidence": 1.0,
            "draft_reply": "Your inquiry has been escalated to a human agent.",
            "grounding_note": "none",
            "decision": "escalate",
            "reason": "Trivial policy: always escalate 100% of inquiries.",
            "max_similarity": 0.0,
            "retrieved_pairs": [],
            "latency_ms": 0.1
        }

class SimpleBaseline:
    """Simple Baseline: TF-IDF classifier + fixed canned reply + escalate only if 'other'."""
    
    def __init__(self):
        self.classifier = BaselineTfidfClassifier()
        self.canned_reply = (
            "Thanks for reaching out to Amazon Help! Please send us a direct message "
            "with your order details so we can investigate this for you."
        )
        
    def run(self, customer_msg: str) -> Dict[str, Any]:
        pred = self.classifier.classify(customer_msg)
        intent = pred["intent"]
        confidence = pred["confidence"]
        
        # Naive rule: escalate only on 'other'
        if intent == "other":
            decision = "escalate"
            reason = "Simple baseline rule: escalate only if intent is 'other'"
        else:
            decision = "auto"
            reason = f"Simple baseline rule: auto-handle non-other intent ({intent})"
            
        return {
            "customer_msg": customer_msg,
            "intent": intent,
            "confidence": confidence,
            "draft_reply": self.canned_reply,
            "grounding_note": "none (canned static template)",
            "decision": decision,
            "reason": reason,
            "max_similarity": 0.0,
            "retrieved_pairs": [],
            "latency_ms": 2.5
        }
