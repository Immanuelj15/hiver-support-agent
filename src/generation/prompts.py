"""
src/generation/prompts.py

System and user prompt templates for evidence-grounded response generation.
Strictly constrains the agent to facts in retrieved evidence, preventing hallucinated
order numbers, carrier guarantees, or unauthorized financial commitments.
"""

from typing import List, Dict, Any

GROUNDED_SYSTEM_PROMPT = """You are a helpful, professional, and empathetic customer support agent for Amazon on Twitter (@AmazonHelp).

RULES:
1. Ground your response STRICTLY in the provided historical evidence. Do not invent tracking information, order numbers, delivery dates, or policies not present in evidence.
2. If an issue requires private personal information (order ID, email, credit card), instruct the customer to reach out via secure direct message or provided support link [link].
3. NEVER make unverified financial promises (e.g. do NOT say "I have refunded your $50" or "Sending you a replacement today" unless instructed).
4. Tone: Empathetic, concise, polite, solution-oriented.
5. Format: Keep replies under 75 words (suitable for customer support Twitter replies).
6. When relevant, reference actions from historical examples without copying user-specific private data.
"""

def format_grounded_user_prompt(
    customer_msg: str,
    intent: str,
    evidences: List[Dict[str, Any]]
) -> str:
    """Format prompt with retrieved historical evidence cases."""
    evidence_block = ""
    for idx, ev in enumerate(evidences, 1):
        conv_id = ev.get("conversation_id", f"case-{idx}")
        cust = ev.get("customer_msg", "").strip()
        reply = ev.get("agent_reply", "").strip()
        evidence_block += f"[EVIDENCE #{idx}] (Ref: {conv_id})\nCustomer: {cust}\nAgent Resolution: {reply}\n\n"

    prompt = f"""Retrieved Historical Resolved Cases:
{evidence_block.strip() if evidence_block else "No prior cases available."}

Current Customer Inquiry:
Customer Message: "{customer_msg}"
Classified Intent: {intent}

Generate a concise, empathetic, and grounded support reply addressing the customer's issue:
Agent Reply:"""
    return prompt

def format_escalation_reply(
    customer_msg: str,
    intent: str,
    reason: str
) -> str:
    """Generate professional handoff reply when escalation to human specialist is required."""
    if intent == "account_security_and_login":
        return (
            "We take your account security very seriously. Please do not share sensitive details publicly. "
            "To safeguard your account immediately, please visit our secure recovery portal here: [link] or contact our "
            "specialist fraud team directly so a senior security representative can assist you."
        )
    elif "safety" in reason.lower() or "hazard" in reason.lower() or "injury" in reason.lower():
        return (
            "We are deeply concerned to hear this and apologize for this distressing experience. "
            "Your safety is our top priority. Please do not handle any damaged or hazardous items. "
            "We are escalating your report to our specialist safety team right away; please send us a DM with your order details."
        )
    elif "legal" in reason.lower() or "chargeback" in reason.lower():
        return (
            "We sincerely apologize for this frustration and want to ensure this is resolved properly. "
            "I am routing your request directly to a senior support supervisor who has full account authorization "
            "to review your order and resolution history. Please send us a secure direct message: [link]."
        )
    else:
        return (
            "We apologize for the inconvenience you are experiencing. Because your issue requires specialized "
            "account review, I am routing this conversation to a human support specialist who can look directly into "
            "your account details. Please send us a private message here: [link] to connect."
        )
