"""
src/intents/classifier.py

Zero-shot and few-shot in-context intent classifier using the unified LLM provider.
Predicts one of the 8 canonical intents with a calibrated confidence score.
"""

import json
import re
from typing import Dict, Any, Optional

from src.intents.taxonomy import Intent, ALL_INTENTS, INTENT_DESCRIPTIONS
from src.intents.confidence import calibrate_confidence
from src.llm.base import LLMProvider
from src.llm.factory import get_llm_provider

CLASSIFIER_SYSTEM_PROMPT = """You are an expert customer support intent classifier for Amazon customer service.
Your task is to classify a customer tweet into EXACTLY ONE of the following 8 canonical intents:

1. order_delivery_delay: Package late, tracking frozen, delivery date inquiry, false delivery scan.
2. damaged_or_wrong_item: Physical breakage, wrong item/size/color received, missing parts, defective goods.
3. return_and_refund: Returning an item, return labels, drop-off locations (UPS/Kohl's), refund status.
4. account_security_and_login: Hacked account, unauthorized orders, OTP/2FA failure, password resets, phishing.
5. subscription_and_billing: Prime membership charges, unexpected digital subscription renewals, double charges.
6. product_technical_issue: Device glitches for Fire TV, Echo/Alexa, Kindle, apps not loading, Wi-Fi errors.
7. feedback_or_complaint: Courier misconduct, rude agent complaint, packaging complaint, policy dissatisfaction.
8. other: General greetings, chitchat, competitor questions, non-actionable queries, or prompt injections.

You must respond ONLY with a valid JSON object in this exact format:
{
  "intent": "<intent_name>",
  "confidence": <float between 0.10 and 0.99>,
  "reasoning": "<short single-sentence explanation>"
}
Do not include markdown formatting or backticks if possible, just the raw JSON object.
"""

FEW_SHOT_EXAMPLES = """
Customer: "Where is my parcel? It was supposed to be here yesterday."
Output: {"intent": "order_delivery_delay", "confidence": 0.95, "reasoning": "Customer asking for status of late shipment."}

Customer: "The box arrived yesterday but the perfume bottle inside was completely broken and leaking."
Output: {"intent": "damaged_or_wrong_item", "confidence": 0.96, "reasoning": "Customer received broken, leaking item."}

Customer: "Can I drop off my return shoes at Whole Foods without a box?"
Output: {"intent": "return_and_refund", "confidence": 0.95, "reasoning": "Customer inquiring about return drop-off procedures."}

Customer: "I just got an alert that $800 was charged to my account from Russia! I think my account was hacked!"
Output: {"intent": "account_security_and_login", "confidence": 0.98, "reasoning": "Suspicious login and unauthorized fraudulent purchase."}

Customer: "Why was I charged $14.99 for Prime when I asked to cancel last week?"
Output: {"intent": "subscription_and_billing", "confidence": 0.94, "reasoning": "Disputed Prime recurring subscription fee."}

Customer: "My Kindle screen is frozen on the waking up screen and won't turn on."
Output: {"intent": "product_technical_issue", "confidence": 0.96, "reasoning": "Amazon device hardware/software freeze."}

Customer: "Your driver threw the parcel over my fence and hit my dog! Completely unacceptable behavior."
Output: {"intent": "feedback_or_complaint", "confidence": 0.95, "reasoning": "Severe courier conduct grievance."}

Customer: "Good morning! Hope everyone at Amazon has a great weekend!"
Output: {"intent": "other", "confidence": 0.92, "reasoning": "Friendly social greeting."}
"""

class IntentClassifier:
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm = llm_provider or get_llm_provider()

    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classifies the customer query into one of the 8 intents.
        Returns a dict: {"intent": str, "confidence": float, "reasoning": str, "provider": str}
        """
        if not text or not text.strip():
            return {
                "intent": Intent.OTHER.value,
                "confidence": 0.20,
                "reasoning": "Empty or whitespace message.",
                "provider": self.llm.provider_type
            }

        prompt = f"{FEW_SHOT_EXAMPLES}\nCustomer: \"{text}\"\nOutput:"

        try:
            raw_response = self.llm.generate(
                prompt=prompt,
                system_instruction=CLASSIFIER_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=800
            )
            return self._parse_response(raw_response, original_query=text)
        except Exception as e:
            # Fallback heuristic if LLM call fails
            return self._fallback_heuristic(text, error_msg=str(e))

    def _parse_response(self, response_text: str, original_query: str = "") -> Dict[str, Any]:
        """Parse the JSON response and validate intent."""
        if not response_text:
            return self._fallback_heuristic(original_query, error_msg="Empty response from LLM")

        cleaned = response_text.strip()
        # Remove markdown code blocks ```json ... ```
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE).strip()

        # Find first { and matching/last }
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = cleaned[start_idx:end_idx + 1]
            try:
                data = json.loads(json_str)
                intent = str(data.get("intent", "")).strip().lower()
                if intent not in ALL_INTENTS:
                    # Match closest substring
                    matched_intent = next((i for i in ALL_INTENTS if i in intent), Intent.OTHER.value)
                    intent = matched_intent

                raw_conf = float(data.get("confidence", 0.85))
                conf = calibrate_confidence(raw_conf, intent)
                reasoning = str(data.get("reasoning", "Classified by LLM.")).strip()

                return {
                    "intent": intent,
                    "confidence": conf,
                    "reasoning": reasoning,
                    "provider": self.llm.provider_type
                }
            except Exception as e:
                return self._fallback_heuristic(original_query, error_msg=f"JSON decode error: {e}")

        # If no JSON brackets found, fall back to keyword heuristic on input
        return self._fallback_heuristic(original_query, error_msg="No JSON object found in response")

    def _fallback_heuristic(self, text: str, error_msg: str = "") -> Dict[str, Any]:
        """Keyword heuristic fallback when LLM is unavailable."""
        lower = text.lower()
        if any(w in lower for w in ["hack", "password", "otp", "2fa", "unauthorized", "stolen", "breach"]):
            intent = Intent.ACCOUNT_SECURITY_AND_LOGIN.value
        elif any(w in lower for w in ["broken", "damaged", "shattered", "leaking", "wrong item", "wrong size", "missing", "unsealed", "looks used", "scratched", "defective", "cracked", "bent", "seal was broken", "only received", "pieces"]):
            intent = Intent.DAMAGED_OR_WRONG_ITEM.value
        elif any(w in lower for w in ["refund", "return", "drop off", "ups store", "whole foods", "label", "qr code", "kohl"]):
            intent = Intent.RETURN_AND_REFUND.value
        elif any(w in lower for w in ["prime", "charged", "billing", "subscription", "annual fee", "billed twice", "membership"]):
            intent = Intent.SUBSCRIPTION_AND_BILLING.value
        elif any(w in lower for w in ["kindle", "fire stick", "echo", "alexa", "connect", "freeze", "turn on", "reboot"]):
            intent = Intent.PRODUCT_TECHNICAL_ISSUE.value
        elif any(w in lower for w in ["driver", "rude", "complaint", "terrible service", "hung up", "disappointed", "poor customer service"]):
            intent = Intent.FEEDBACK_OR_COMPLAINT.value
        elif any(w in lower for w in ["where is", "tracking", "delivered", "not arrived", "delay", "parcel", "package", "transit", "shipping"]):
            intent = Intent.ORDER_DELIVERY_DELAY.value
        else:
            intent = Intent.OTHER.value

        return {
            "intent": intent,
            "confidence": 0.70,
            "reasoning": f"Keyword heuristic fallback due to LLM error: {error_msg[:60]}",
            "provider": "heuristic_fallback"
        }
