"""
src/reply_agent.py

Drafts customer support replies grounded in historical resolution pairs.
Infers brand tone, brevity, and policy from retrieved examples.
Outputs {draft_reply, grounding_note, retrieved_similarity_max}.
"""

import os
import sys
import re
import json
import argparse
from typing import Dict, Any, List
from pathlib import Path

# Ensure repo root in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import LLM_MODEL, get_llm_client

class ReplyAgent:
    """RAG-grounded customer service reply generator."""
    
    def __init__(self, model_name: str = None, provider: str = None):
        self.model_name = model_name or LLM_MODEL
        self.client = get_llm_client(provider=provider)
        
    def draft_reply(
        self,
        customer_msg: str,
        intent: str,
        retrieved_pairs: List[Dict[str, Any]],
        max_similarity: float
    ) -> Dict[str, Any]:
        """
        Generates a reply grounded in the retrieved historical pairs.
        """
        # Format evidence block
        evidence_block = ""
        for i, pair in enumerate(retrieved_pairs, 1):
            evidence_block += (
                f"\n--- Historical Case {i} (Similarity: {pair.get('similarity', 0.0):.3f}) ---\n"
                f"Customer Query: {pair.get('customer_msg', '')}\n"
                f"Amazon's Actual Resolution: {pair.get('brand_reply', '')}\n"
            )
            
        system_prompt = (
            "You are an AI customer support agent representing Amazon customer care on social media.\n"
            "Your objective is to draft an empathetic, professional, and helpful response to an incoming customer message.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. GROUNDING: Ground your response strictly in the provided historical resolution cases. Infer the brand's "
            "typical tone, phrasing, conciseness, and action steps from these cases. Do not invent company policies.\n"
            "2. PRIVACY & SAFETY: Never ask the customer to post sensitive personal information (credit cards, passwords) publicly. "
            "If account verification is required, advise them to send a direct message (DM) with their order ID or use the secure help link.\n"
            "3. GROUNDING NOTE: You must state which Case(s) (e.g., 'Case 1', 'Case 1 & 2') you relied on as the basis for your reply, "
            "or 'none' if the historical cases were not relevant and you had to improvise.\n"
            "4. OUTPUT FORMAT: Return valid JSON ONLY with schema:\n"
            "   {\n"
            "     \"draft_reply\": \"<your drafted response text>\",\n"
            "     \"grounding_note\": \"<which case(s) you based this on and why, or 'none'>\"\n"
            "   }\n"
            "5. Do NOT include markdown ticks or text outside the JSON object."
        )
        
        user_prompt = (
            f"Customer Message: {customer_msg}\n"
            f"Detected Intent: {intent}\n\n"
            f"Retrieved Historical Resolutions for Reference:\n"
            f"{evidence_block if evidence_block else 'None available.'}\n\n"
            "Draft the response now in JSON format."
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            raw_content = response.choices[0].message.content.strip()
            
            clean_json = raw_content
            if clean_json.startswith("```"):
                clean_json = re.sub(r"^```(?:json)?\s*", "", clean_json)
                clean_json = re.sub(r"\s*```$", "", clean_json)
                
            data = json.loads(clean_json)
            draft_reply = data.get("draft_reply", "").strip()
            grounding_note = data.get("grounding_note", "none").strip()
            
            return {
                "draft_reply": draft_reply,
                "grounding_note": grounding_note,
                "retrieved_similarity_max": max_similarity
            }
        except Exception as e:
            # Fallback safe template
            return {
                "draft_reply": "We apologize for the inconvenience with your order. Please reach out to us via direct message with your order details so we can investigate further.",
                "grounding_note": f"none (fallback triggered by error: {str(e)})",
                "retrieved_similarity_max": max_similarity
            }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", type=str, default="My delivery is missing but tracking says it was left at the porch.")
    args = parser.parse_args()
    
    from src.retriever import HistoricalRetriever
    retriever = HistoricalRetriever()
    pairs, max_sim = retriever.retrieve(args.message, intent="order_delivery_delay")
    
    agent = ReplyAgent()
    res = agent.draft_reply(args.message, intent="order_delivery_delay", retrieved_pairs=pairs, max_similarity=max_sim)
    print("\n--- Reply Agent Output ---")
    print(f"Customer Message: {args.message}")
    print(f"Max Retrieval Similarity: {res['retrieved_similarity_max']}")
    print(f"Grounding Note: {res['grounding_note']}")
    print(f"Draft Reply: {res['draft_reply']}\n")
