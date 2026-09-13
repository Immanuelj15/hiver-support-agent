"""
eval/label_cli.py

Interactive CLI tool to browse, label, and verify golden set examples.
Supports stratified sampling from threads.jsonl and appending hand-annotated entries to golden_150.jsonl.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Ensure repo root in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import INTENTS, GOLDEN_SET_PATH, THREADS_PATH

def run_label_cli(threads_path: str = str(THREADS_PATH), output_path: str = str(GOLDEN_SET_PATH)):
    print("\n" + "="*80)
    print("GOLDEN SET INTERACTIVE LABELING CLI")
    print("="*80)
    print("Available Intents:")
    for idx, intent in enumerate(INTENTS, 1):
        print(f"  {idx}. {intent}")
    print("\nPress Ctrl+C or type 'exit' at any prompt to quit and save.\n")
    
    # Load existing labels to avoid duplicates
    existing_queries = set()
    existing_count = 0
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    existing_queries.add(item["customer_msg"].strip())
                    existing_count += 1
    print(f"Existing labeled examples in {output_path}: {existing_count}")
    
    # Read candidate threads
    candidates = []
    with open(threads_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                msg = item.get("customer_msg", "").strip()
                if msg and msg not in existing_queries:
                    candidates.append(item)
                    
    print(f"Candidate messages available for labeling: {len(candidates):,}\n")
    
    out_file = open(output_path, "a", encoding="utf-8")
    
    labeled_this_session = 0
    try:
        for item in candidates:
            msg = item["customer_msg"]
            brand_reply = item["brand_reply"]
            
            print("-" * 80)
            print(f"Customer Message: {msg}")
            print(f"Historical Amazon Reply: {brand_reply}")
            print("-" * 80)
            
            # Select Intent
            while True:
                intent_input = input(f"Enter Intent [1-{len(INTENTS)}] or name (s to skip, q to quit): ").strip()
                if intent_input.lower() in ("q", "exit"):
                    return
                if intent_input.lower() == "s":
                    break
                if intent_input.isdigit() and 1 <= int(intent_input) <= len(INTENTS):
                    chosen_intent = INTENTS[int(intent_input) - 1]
                    break
                elif intent_input in INTENTS:
                    chosen_intent = intent_input
                    break
                else:
                    print("Invalid choice, please re-enter.")
            if intent_input.lower() == "s":
                continue
                
            # Escalation Decision
            while True:
                esc_input = input("Should Escalate? [y/n]: ").strip().lower()
                if esc_input in ("y", "yes"):
                    should_escalate = True
                    break
                elif esc_input in ("n", "no"):
                    should_escalate = False
                    break
                elif esc_input in ("q", "exit"):
                    return
                else:
                    print("Please enter 'y' or 'n'.")
                    
            # Escalation Reason Notes
            esc_reason = input("Escalation Reason / Notes (e.g. sensitive, low-info, safe-auto): ").strip()
            
            # Reference reply
            use_hist = input("Use historical reply as reference? [Y/n]: ").strip().lower()
            if use_hist in ("", "y", "yes"):
                ref_reply = brand_reply
            else:
                ref_reply = input("Enter custom reference reply (or leave blank for null): ").strip() or None
                
            record = {
                "id": existing_count + labeled_this_session + 1,
                "customer_msg": msg,
                "gold_intent": chosen_intent,
                "gold_should_escalate": should_escalate,
                "gold_reason_notes": esc_reason,
                "reference_reply": ref_reply
            }
            
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_file.flush()
            labeled_this_session += 1
            print(f"Saved! (Total this session: {labeled_this_session}, Grand Total: {existing_count + labeled_this_session})\n")
            
    except KeyboardInterrupt:
        print("\nSession interrupted by user.")
    finally:
        out_file.close()
        print(f"Finished session. Total labeled in this session: {labeled_this_session}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads_path", type=str, default=str(THREADS_PATH))
    parser.add_argument("--output_path", type=str, default=str(GOLDEN_SET_PATH))
    args = parser.parse_args()
    
    run_label_cli(threads_path=args.threads_path, output_path=args.output_path)
