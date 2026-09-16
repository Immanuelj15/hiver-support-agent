"""
src/evaluation/llm_judge.py

LLM-as-a-Judge implementation for evaluating customer support replies.
Scores responses across the 6 canonical dimensions on a 1 to 5 scale:
Groundedness, Correctness, Relevance, Helpfulness, Tone, and Resolution.
"""

import json
import re
from typing import Dict, Any, Optional

from src.llm.base import LLMProvider
from src.llm.factory import get_llm_provider

JUDGE_SYSTEM_PROMPT = """You are an impartial, senior customer support quality assurance auditor evaluating an AI assistant's reply for Amazon customer service.

Score the candidate response from 1 (terrible) to 5 (flawless) on EACH of the following 6 dimensions:
1. groundedness: Does the reply stick strictly to verifiable facts, without hallucinating fake order numbers, delivery dates, or policies?
2. correctness: Is the advice completely compliant with standard Amazon support procedures?
3. relevance: Does the reply directly answer the customer's specific inquiry?
4. helpfulness: Does the reply provide clear self-serve action paths (e.g. [link], tracking, return instructions)?
5. tone: Is the response empathetic, professional, calm, and respectful?
6. resolution: Does the reply provide an unambiguous next step or safe specialist routing?

You must respond ONLY with a valid JSON object in this format:
{
  "groundedness": <1-5>,
  "correctness": <1-5>,
  "relevance": <1-5>,
  "helpfulness": <1-5>,
  "tone": <1-5>,
  "resolution": <1-5>,
  "rationale": "<short 1-2 sentence evaluation summary>"
}
"""

class LLMJudge:
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm = llm_provider or get_llm_provider()

    def evaluate(
        self,
        customer_msg: str,
        candidate_reply: str,
        gold_intent: str,
        reference_reply: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluates candidate response across 6 dimensions."""
        ref_context = f"\nHuman Reference Reply: \"{reference_reply}\"" if reference_reply else ""
        prompt = f"""Customer Inquiry: "{customer_msg}"
Issue Intent: {gold_intent}{ref_context}

Candidate Agent Reply to Evaluate:
"{candidate_reply}"

Evaluate the candidate reply and return the JSON score object:"""

        try:
            raw = self.llm.generate(
                prompt=prompt,
                system_instruction=JUDGE_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=600
            )
            return self._parse_scores(raw)
        except Exception as e:
            return self._heuristic_scores(candidate_reply, gold_intent, error=str(e))

    def _parse_scores(self, raw_text: str) -> Dict[str, Any]:
        cleaned = raw_text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE).strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            try:
                data = json.loads(cleaned[start:end + 1])
                scores = {}
                for dim in ["groundedness", "correctness", "relevance", "helpfulness", "tone", "resolution"]:
                    val = float(data.get(dim, 4.0))
                    scores[dim] = max(1.0, min(5.0, val))
                scores["rationale"] = str(data.get("rationale", "Evaluated by LLM Judge.")).strip()
                return scores
            except Exception:
                pass

        return self._heuristic_scores(cleaned, "unknown", error="JSON parse failure")

    def _heuristic_scores(self, reply: str, intent: str, error: str = "") -> Dict[str, Any]:
        """Deterministic heuristic fallback if judge call fails."""
        has_link = "[link]" in reply or "http" in reply
        has_empathy = any(w in reply.lower() for w in ["sorry", "apologize", "understand", "concern"])
        base = 4.0 if (has_link and has_empathy) else 3.5

        return {
            "groundedness": base,
            "correctness": base,
            "relevance": base,
            "helpfulness": 4.0 if has_link else 3.0,
            "tone": 4.5 if has_empathy else 3.5,
            "resolution": 4.0 if has_link else 3.0,
            "rationale": f"Heuristic fallback evaluation ({error[:50]})."
        }
