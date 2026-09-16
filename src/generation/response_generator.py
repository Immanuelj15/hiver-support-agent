"""
src/generation/response_generator.py

Generates grounded support replies using LLMProvider and retrieved evidence.
"""

from typing import List, Dict, Any, Optional
from src.llm.base import LLMProvider
from src.llm.factory import get_llm_provider
from src.generation.prompts import (
    GROUNDED_SYSTEM_PROMPT,
    format_grounded_user_prompt,
    format_escalation_reply
)

class ResponseGenerator:
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm = llm_provider or get_llm_provider()

    def generate(
        self,
        customer_msg: str,
        intent: str,
        evidences: List[Dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 500
    ) -> str:
        """
        Generates grounded customer support reply citing retrieved evidence.
        """
        prompt = format_grounded_user_prompt(customer_msg, intent, evidences)

        try:
            reply = self.llm.generate(
                prompt=prompt,
                system_instruction=GROUNDED_SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return reply.strip()
        except Exception as e:
            # Safe fallback if generation fails
            return (
                "We apologize for the inconvenience! For immediate assistance with your order or account, "
                "please reach us directly via our customer support portal here: [link]."
            )

    def generate_escalation_reply(
        self,
        customer_msg: str,
        intent: str,
        reason: str
    ) -> str:
        """Generates deterministic empathetic handoff message for human specialist review."""
        return format_escalation_reply(customer_msg, intent, reason)
