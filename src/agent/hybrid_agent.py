"""
src/agent/hybrid_agent.py

Dynamic Two-Tier Hybrid Architecture for Hiver Customer Support Agent.
Intelligently routes customer inquiries between:
- Tier 1: Local Offline LLM (Ollama / Mistral) for high-confidence, safe, routine queries (Zero Cloud Cost)
- Tier 2: Cloud High-Capability LLM (Groq / Gemini) for ambiguous, out-of-domain, or sensitive queries
"""

import time
from typing import Dict, Any, Optional, List
from pathlib import Path

from src.agent.support_agent import SupportAgent
from src.intents.classifier import IntentClassifier
from src.retrieval.retriever import SupportCaseRetriever
from src.escalation.policy import EscalationPolicy
from src.escalation.decision import EscalationAction, RiskLevel
from src.generation.response_generator import ResponseGenerator
from src.llm.base import LLMProvider
from src.llm.ollama_provider import OllamaProvider
from src.llm.cloud_provider import CloudLLMProvider
from src.llm.factory import get_llm_provider

class HybridSupportAgent(SupportAgent):
    def __init__(
        self,
        local_llm: Optional[LLMProvider] = None,
        cloud_llm: Optional[LLMProvider] = None,
        retriever: Optional[SupportCaseRetriever] = None,
        escalation_policy: Optional[EscalationPolicy] = None,
        local_confidence_threshold: float = 0.85,
        local_similarity_threshold: float = 0.65,
        top_k: int = 3
    ):
        # Local provider (default: Ollama mistral)
        try:
            self.local_llm = local_llm or OllamaProvider(model="mistral:latest")
        except Exception:
            self.local_llm = None

        # Cloud provider (default: Groq or Gemini)
        try:
            self.cloud_llm = cloud_llm or CloudLLMProvider(provider_name="groq")
        except Exception:
            self.cloud_llm = get_llm_provider()

        # Active default provider for parent components
        super().__init__(
            llm_provider=self.cloud_llm,
            retriever=retriever,
            escalation_policy=escalation_policy,
            top_k=top_k
        )

        self.local_confidence_threshold = local_confidence_threshold
        self.local_similarity_threshold = local_similarity_threshold

        # Dedicated generators for each tier
        if self.local_llm:
            self.local_generator = ResponseGenerator(llm_provider=self.local_llm)
        else:
            self.local_generator = self.generator
        self.cloud_generator = ResponseGenerator(llm_provider=self.cloud_llm)

    def route_decision(
        self,
        intent: str,
        confidence: float,
        similarity: float,
        should_escalate: bool,
        risk_level: RiskLevel
    ) -> str:
        """
        Determines whether to route to 'local' (Ollama) or 'cloud' (Groq/Gemini).
        Routing policy:
        - If query is safe (LOW risk), not escalated, with high confidence and high retrieval grounding -> LOCAL
        - Otherwise (ambiguous, complex, high-risk, low confidence) -> CLOUD
        """
        if not self.local_llm:
            return "cloud"

        is_high_confidence = (confidence >= self.local_confidence_threshold)
        is_well_grounded = (similarity >= self.local_similarity_threshold)
        is_safe = (risk_level == RiskLevel.LOW and not should_escalate)

        if is_high_confidence and is_well_grounded and is_safe:
            return "local"
        return "cloud"

    def process(self, customer_msg: str) -> Dict[str, Any]:
        """
        Processes query with dynamic hybrid routing and full audit telemetry.
        """
        start_time = time.perf_counter()

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

        # 4. Dynamic Hybrid Routing
        routed_tier = self.route_decision(
            intent=intent,
            confidence=confidence,
            similarity=max_sim,
            should_escalate=decision.should_escalate,
            risk_level=decision.risk_level
        )

        active_generator = self.local_generator if routed_tier == "local" else self.cloud_generator
        active_llm = self.local_llm if routed_tier == "local" else self.cloud_llm

        # 5. Response generation
        generation_start = time.perf_counter()
        if decision.action == EscalationAction.AUTO_HANDLE:
            try:
                draft_reply = active_generator.generate(
                    customer_msg=customer_msg,
                    intent=intent,
                    evidences=evidences
                )
            except Exception:
                # Fallback to cloud if local fails
                routed_tier = "cloud (fallback)"
                draft_reply = self.cloud_generator.generate(
                    customer_msg=customer_msg,
                    intent=intent,
                    evidences=evidences
                )

            # Post-generation compliance check
            financial_violation = self.escalation_policy.validate_reply(draft_reply)
            if financial_violation:
                decision.action = EscalationAction.ESCALATE
                decision.should_escalate = True
                decision.reason = financial_violation
                decision.risk_level = RiskLevel.HIGH
                draft_reply = self.cloud_generator.generate_escalation_reply(
                    customer_msg=customer_msg,
                    intent=intent,
                    reason=financial_violation
                )
        else:
            draft_reply = active_generator.generate_escalation_reply(
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

        total_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

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
            "routed_tier": routed_tier,
            "cloud_call_avoided": (1 if routed_tier == "local" else 0),
            "provider": active_llm.provider_type if active_llm else "unknown",
            "model": active_llm.model_name if active_llm else "unknown",
            "latency_ms": total_latency_ms
        }

