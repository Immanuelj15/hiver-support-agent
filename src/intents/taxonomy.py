"""
src/intents/taxonomy.py

Defines the 8 empirical customer-support intents and their behavioral configurations.
"""

from enum import Enum
from typing import List, Dict, Any

class Intent(str, Enum):
    ORDER_DELIVERY_DELAY = "order_delivery_delay"
    DAMAGED_OR_WRONG_ITEM = "damaged_or_wrong_item"
    RETURN_AND_REFUND = "return_and_refund"
    ACCOUNT_SECURITY_AND_LOGIN = "account_security_and_login"
    SUBSCRIPTION_AND_BILLING = "subscription_and_billing"
    PRODUCT_TECHNICAL_ISSUE = "product_technical_issue"
    FEEDBACK_OR_COMPLAINT = "feedback_or_complaint"
    OTHER = "other"

ALL_INTENTS: List[str] = [intent.value for intent in Intent]

SENSITIVE_INTENTS: List[str] = [
    Intent.ACCOUNT_SECURITY_AND_LOGIN.value,
    Intent.OTHER.value
]

INTENT_DESCRIPTIONS: Dict[str, str] = {
    Intent.ORDER_DELIVERY_DELAY.value: "Package late, tracking frozen, false delivery scan, or transit inquiry.",
    Intent.DAMAGED_OR_WRONG_ITEM.value: "Physical defect, broken seals, wrong size/color, missing parts, or damaged goods.",
    Intent.RETURN_AND_REFUND.value: "Return label generation, drop-off location assistance, or status of expected refund.",
    Intent.ACCOUNT_SECURITY_AND_LOGIN.value: "Compromised account, OTP failures, 2FA recovery, password reset, or suspicious charges.",
    Intent.SUBSCRIPTION_AND_BILLING.value: "Unauthorized Prime membership renewal, digital charges, or recurring subscription fees.",
    Intent.PRODUCT_TECHNICAL_ISSUE.value: "Hardware/software troubleshooting for Kindle, Fire TV, Echo/Alexa, or Amazon apps.",
    Intent.FEEDBACK_OR_COMPLAINT.value: "Driver misconduct, packaging complaints, customer service grievances, or policy venting.",
    Intent.OTHER.value: "General greetings, ambiguous queries, out-of-scope requests, prompt injections, or chitchat."
}
