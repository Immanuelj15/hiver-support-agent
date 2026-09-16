"""
tests/test_agent.py

Integration tests for the SupportAgent pipeline coordinating classification,
retrieval, escalation, and response generation.
"""

import pytest
from src.agent.support_agent import SupportAgent

class MockAgentLLM:
    def __init__(self):
        self.model_name = "mock-agent-llm"
        self.provider_type = "mock"

    def generate(self, prompt, system_instruction=None, **kwargs):
        # If classifier prompt
        if "classifier" in (system_instruction or "").lower() or "classify" in (system_instruction or "").lower():
            # Check the trailing lines of the prompt for the target inquiry
            trailing_text = "\n".join(prompt.strip().split("\n")[-3:]).lower()
            if any(w in trailing_text for w in ["hacked", "stolen", "security", "card"]):
                return '{"intent": "account_security_and_login", "confidence": 0.98, "reasoning": "Security breach."}'
            else:
                return '{"intent": "order_delivery_delay", "confidence": 0.95, "reasoning": "Delay inquiry."}'
        # If response generator prompt
        return "I apologize for the delay! You can check your order tracking at [link]."

def test_support_agent_auto_handle():
    mock_llm = MockAgentLLM()
    agent = SupportAgent(llm_provider=mock_llm)

    res = agent.process("Where is my package? It was supposed to be here yesterday.")
    assert "customer_msg" in res
    assert "intent" in res
    assert "action" in res
    assert "should_escalate" in res
    assert "reply" in res
    assert res["action"] == "AUTO_HANDLE"
    assert res["should_escalate"] is False
    assert "[link]" in res["reply"]

def test_support_agent_escalation():
    mock_llm = MockAgentLLM()
    agent = SupportAgent(llm_provider=mock_llm)

    res = agent.process("My account was hacked and someone bought items with my card!")
    assert res["intent"] == "account_security_and_login"
    assert res["action"] == "ESCALATE"
    assert res["should_escalate"] is True
    assert "security" in res["escalation_reason"].lower()
    assert "recovery" in res["reply"].lower() or "specialist" in res["reply"].lower()
