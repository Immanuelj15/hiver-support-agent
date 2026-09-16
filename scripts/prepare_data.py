"""
scripts/prepare_data.py

Prepares the clean knowledge corpus for retrieval (data/processed/knowledge_corpus.jsonl)
by taking ingested threads, filtering out any evaluation/golden set overlap,
and enforcing strict conversation-level leakage checks.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_jsonl, save_jsonl
from src.data.cleaner import clean_tweet_text
from src.data.splitter import check_leakage

def main():
    parser = argparse.ArgumentParser(description="Prepare knowledge corpus from processed threads.")
    parser.add_argument("--threads", default="data/processed/threads.jsonl", help="Path to threads.jsonl")
    parser.add_argument("--golden", default="data/golden/golden_set.jsonl", help="Path to golden_set.jsonl")
    parser.add_argument("--output", default="data/processed/knowledge_corpus.jsonl", help="Output corpus path")
    parser.add_argument("--max-cases", type=int, default=10000, help="Maximum historical cases to index")
    args = parser.parse_args()

    print(f"Loading threads from {args.threads}...")
    threads = load_jsonl(args.threads)
    print(f"Loaded {len(threads):,} total threads.")

    # Load golden set to blacklist
    golden_items = []
    if os.path.exists(args.golden):
        golden_items = load_jsonl(args.golden)
        print(f"Loaded {len(golden_items)} golden evaluation items for leakage filtering.")

    golden_msgs = {item["customer_msg"].strip().lower() for item in golden_items if "customer_msg" in item}

    # Filter threads
    clean_corpus = []
    skipped_leakage = 0
    skipped_invalid = 0

    for item in threads:
        c_msg = clean_tweet_text(item.get("customer_msg", ""))
        raw_reply = item.get("agent_reply") or item.get("brand_reply", "")
        a_reply = clean_tweet_text(raw_reply, is_brand_reply=True)

        if not c_msg or not a_reply or len(c_msg) < 10 or len(a_reply) < 10:
            skipped_invalid += 1
            continue

        if c_msg.lower() in golden_msgs:
            skipped_leakage += 1
            continue

        thread_id = str(item.get("thread_id") or item.get("id") or len(clean_corpus) + 1)

        clean_corpus.append({
            "conversation_id": thread_id,
            "customer_msg": c_msg,
            "agent_reply": a_reply,
            "intent": item.get("intent", "order_delivery_delay")
        })

        if len(clean_corpus) >= args.max_cases:
            break

    print(f"Filtered {skipped_invalid} invalid/short threads, {skipped_leakage} golden set collisions.")
    print(f"Prepared {len(clean_corpus):,} clean knowledge corpus items.")

    # Leakage verification
    if golden_items:
        is_clean, violations = check_leakage(clean_corpus, golden_items)
        if is_clean:
            print("[PASS] Strict zero-leakage check confirmed! No thread or customer message collisions.")
        else:
            print(f"[FAIL] Warning: Found {len(violations)} leakage violations:")
            for v in violations[:5]:
                print(f"  - {v}")

    # Print summary statistics
    c_lens = [len(x["customer_msg"].split()) for x in clean_corpus]
    r_lens = [len(x["agent_reply"].split()) for x in clean_corpus]
    print("\n--- Knowledge Corpus Summary Statistics ---")
    print(f"Total historical cases: {len(clean_corpus):,}")
    print(f"Customer message word count: median={np.median(c_lens):.1f}, mean={np.mean(c_lens):.1f}, min={np.min(c_lens)}, max={np.max(c_lens)}")
    print(f"Agent reply word count:      median={np.median(r_lens):.1f}, mean={np.mean(r_lens):.1f}, min={np.min(r_lens)}, max={np.max(r_lens)}")

    save_jsonl(clean_corpus, args.output)
    print(f"\nSaved knowledge corpus to {args.output}")

if __name__ == "__main__":
    main()
