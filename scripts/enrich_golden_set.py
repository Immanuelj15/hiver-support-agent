"""
scripts/enrich_golden_set.py

Enriches data/golden/golden_set.jsonl with:
- conversation_id
- gold_escalation_decision (explicit 'AUTO_HANDLE' or 'ESCALATE_TO_HUMAN')
- gold_reason
- difficulty category: 'common', 'rare', 'ambiguous', 'multi_intent', 'sarcastic', 'escalation_hazard'
"""

import json
from pathlib import Path
from collections import Counter

golden_path = Path("data/golden/golden_set.jsonl")

with open(golden_path, "r", encoding="utf-8") as f:
    items = [json.loads(line) for line in f if line.strip()]

for item in items:
    # 1. Ensure string escalation decision
    if "gold_escalation_decision" not in item:
        item["gold_escalation_decision"] = "ESCALATE_TO_HUMAN" if item.get("gold_should_escalate") else "AUTO_HANDLE"
    
    # 2. Ensure gold_reason and conversation_id
    if "gold_reason" not in item:
        item["gold_reason"] = item.get("gold_reason_notes", "")
    if "conversation_id" not in item:
        item["conversation_id"] = f"conv_eval_{item.get('id', 0):04d}"

    # 3. Categorize difficulty
    msg = item["customer_msg"].lower()
    notes = item.get("gold_reason_notes", "").lower()
    intent = item.get("gold_intent", "")

    # Sarcastic / Emotional Venting
    sarcasm_markers = [
        "sarcasm", "sarcastic", "brilliant", "love paying", "outstanding work", 
        "great job", "joke", "pathetic", "ridiculous", "clowns", "worst service",
        "thanks for nothing", "unbelievable", "trash", "useless"
    ]
    # Escalation / Safety / Legal / Security
    hazard_markers = [
        "fire", "smoke", "exploded", "bleeding", "hospital", "hazard", "lawyer", 
        "attorney", "sue", "lawsuit", "hacked", "unauthorized", "fraud", "police", 
        "stolen", "dispute", "chargeback", "court", "bbb", "ftc", "injur"
    ]

    if any(w in msg or w in notes for w in hazard_markers) or item.get("gold_should_escalate") and intent == "account_security_and_login":
        diff = "escalation_hazard"
    elif any(w in msg or w in notes for w in sarcasm_markers):
        diff = "sarcastic"
    elif (" and " in msg or " but " in msg or " also " in msg) and (sum(w in msg for w in ["late", "broken", "damaged", "charged", "refund", "card", "wrong", "missing"]) >= 2):
        diff = "multi_intent"
    elif intent == "other" or len(msg.split()) <= 5 or any(w in msg for w in ["who are you", "what is this", "anyone there", "??", "help me"]):
        diff = "ambiguous"
    elif intent in ["product_technical_issue", "subscription_and_billing"]:
        diff = "rare"
    else:
        diff = "common"

    item["difficulty"] = diff


# Write back
with open(golden_path, "w", encoding="utf-8") as f:
    for item in items:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

counts = Counter([x["difficulty"] for x in items])
print(f"Successfully updated {len(items)} items in {golden_path}")
print("Difficulty distribution:")
for k, v in counts.items():
    print(f"  {k}: {v} ({v/len(items)*100:.1f}%)")
