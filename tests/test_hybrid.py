"""
tests/test_hybrid.py

Unit tests for dynamic two-tier HybridSupportAgent routing,
cloud deflection measurement, and structured audit outputs.
"""

from src.agent.hybrid_agent import HybridSupportAgent
from src.escalation.decision import RiskLevel

class MockLLM:
    def __init__(self, provider_type: str, model_name: str):
        self.provider_type = provider_type
        self.model_name = model_name

    def generate(self, prompt: str, **kwargs) -> str:
        return f"Mock response from {self.provider_type} ({self.model_name})"

def test_hybrid_routing_decision():
    local_mock = MockLLM("ollama", "mistral:latest")
    cloud_mock = MockLLM("cloud", "openai/gpt-oss-20b")

    agent = HybridSupportAgent(
        local_llm=local_mock,
        cloud_llm=cloud_mock,
        local_confidence_threshold=0.85,
        local_similarity_threshold=0.65
    )

    # 1. High confidence, high similarity, low risk -> LOCAL
    tier_safe = agent.route_decision(
        intent="order_delivery_delay",
        confidence=0.92,
        similarity=0.78,
        should_escalate=False,
        risk_level=RiskLevel.LOW
    )
    assert tier_safe == "local"

    # 2. Low confidence -> CLOUD
    tier_low_conf = agent.route_decision(
        intent="order_delivery_delay",
        confidence=0.60,
        similarity=0.80,
        should_escalate=False,
        risk_level=RiskLevel.LOW
    )
    assert tier_low_conf == "cloud"

    # 3. High risk / Escalation -> CLOUD
    tier_hazard = agent.route_decision(
        intent="damaged_or_wrong_item",
        confidence=0.95,
        similarity=0.85,
        should_escalate=True,
        risk_level=RiskLevel.CRITICAL
    )
    assert tier_hazard == "cloud"

def test_hybrid_process_output_structure():
    local_mock = MockLLM("ollama", "mistral:latest")
    cloud_mock = MockLLM("cloud", "openai/gpt-oss-20b")

    agent = HybridSupportAgent(
        local_llm=local_mock,
        cloud_llm=cloud_mock
    )

    res = agent.process("Where is my package?")
    assert "routed_tier" in res
    assert res["routed_tier"] in ("local", "cloud", "cloud (fallback)")
    assert "cloud_call_avoided" in res
    assert res["cloud_call_avoided"] in (0, 1)
    assert "action" in res
    assert "retrieved_evidence" in res
    assert "latency_ms" in res
