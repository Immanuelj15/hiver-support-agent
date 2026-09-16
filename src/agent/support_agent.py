"""
src/agent/support_agent.py

Unified customer support agent coordinating intent classification, dense retrieval,
deterministic escalation policy, and grounded response generation.
"""

from typing import Dict, Any, Optional, List

from src.intents.classifier import IntentClassifier
from src.retrieval.retriever import SupportCaseRetriever
from src.escalation.policy import EscalationPolicy
from src.escalation.decision import EscalationAction, RiskLevel
from src.generation.response_generator import ResponseGenerator
from src.llm.base import LLMProvider
from src.llm.factory import get_llm_provider

class SupportAgent:
    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        retriever: Optional[SupportCaseRetriever] = None,
        escalation_policy: Optional[EscalationPolicy] = None,
        top_k: int = 3
    ):
        self.llm = llm_provider or get_llm_provider()
        self.classifier = IntentClassifier(llm_provider=self.llm)
        self.retriever = retriever or SupportCaseRetriever()
        self.escalation_policy = escalation_policy or EscalationPolicy()
        self.generator = ResponseGenerator(llm_provider=self.llm)
        self.top_k = top_k

    def process(self, customer_msg: str) -> Dict[str, Any]:
        """
        Processes a single customer message through the end-to-end agent pipeline.
        Returns a structured dictionary with complete telemetry and audit trail.
        """
        # 1. Intent classification
        intent_res = self.classifier.classify(customer_msg)
        intent = intent_res["intent"]
        confidence = float(intent_res["confidence"])

        # 2. Dense semantic retrieval
        try:
            evidences = self.retriever.retrieve(customer_msg, top_k=self.top_k)
            max_sim = evidences[0]["similarity"] if evidences else 0.0
        except Exception:
            evidences = []
            max_sim = 0.0

        # 3. Deterministic escalation policy evaluation
        decision = self.escalation_policy.evaluate(
            customer_msg=customer_msg,
            predicted_intent=intent,
            intent_confidence=confidence,
            retrieval_similarity=max_sim
        )

        # 4. Response generation
        if decision.action == EscalationAction.AUTO_HANDLE:
            draft_reply = self.generator.generate(
                customer_msg=customer_msg,
                intent=intent,
                evidences=evidences
            )

            # Post-generation compliance check for unverified financial promises
            financial_violation = self.escalation_policy.validate_reply(draft_reply)
            if financial_violation:
                decision.action = EscalationAction.ESCALATE
                decision.should_escalate = True
                decision.reason = financial_violation
                decision.risk_level = RiskLevel.HIGH
                draft_reply = self.generator.generate_escalation_reply(
                    customer_msg=customer_msg,
                    intent=intent,
                    reason=financial_violation
                )
        else:
            draft_reply = self.generator.generate_escalation_reply(
                customer_msg=customer_msg,
                intent=intent,
                reason=decision.reason
            )

        formatted_evidence = [
            {
                "conversation_id": ev.get("conversation_id", ""),
                "customer_message": ev.get("customer_msg", ""),
                "brand_reply": ev.get("agent_reply", ""),
                "similarity": round(float(ev.get("similarity", 0.0)), 4)
            }
            for ev in evidences
        ]

        return {
            "customer_msg": customer_msg,
            "intent": intent,
            "intent_confidence": round(confidence, 4),
            "retrieval_score": round(max_sim, 4),
            "retrieval_similarity": round(max_sim, 4),
            "evidence_count": len(evidences),
            "decision": decision.action.value,
            "decision_reason": decision.reason,
            "action": decision.action.value,
            "should_escalate": decision.should_escalate,
            "escalation_reason": decision.reason,
            "risk_level": decision.risk_level.value,
            "triggered_rule": decision.triggered_rule,
            "reply": draft_reply,
            "evidence": formatted_evidence,
            "retrieved_evidence": evidences,
            "provider": self.llm.provider_type,
            "model": self.llm.model_name
        }

