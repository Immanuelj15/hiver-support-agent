"""
src/data/splitter.py

Performs conversation-level splitting and leakage verification.
Guarantees that test / evaluation conversations are strictly disjoint from the retrieval corpus.
"""

import random
from typing import List, Dict, Any, Tuple, Set

def check_leakage(
    train_corpus: List[Dict[str, Any]],
    eval_set: List[Dict[str, Any]]
) -> Tuple[bool, List[str]]:
    """
    Checks for two forms of leakage:
    1. Conversation ID overlap (same thread in both)
    2. Exact normalized customer message collision
    """
    train_thread_ids: Set[str] = {
        str(t.get("thread_id") or t.get("conversation_id") or t.get("id"))
        for t in train_corpus
        if (t.get("thread_id") or t.get("conversation_id") or t.get("id")) is not None
    }
    
    train_messages: Set[str] = {
        t["customer_msg"].strip().lower()
        for t in train_corpus
        if "customer_msg" in t and t["customer_msg"]
    }
    
    violations = []
    
    for item in eval_set:
        eval_id = str(item.get("thread_id") or item.get("conversation_id") or item.get("id", ""))
        if eval_id and eval_id in train_thread_ids:
            violations.append(f"Thread ID collision: {eval_id}")
            
        msg = item.get("customer_msg", "").strip().lower()
        if msg and msg in train_messages:
            violations.append(f"Duplicate customer message text: '{msg[:60]}...'")
            
    is_clean = (len(violations) == 0)
    return is_clean, violations

def conversation_split(
    threads: List[Dict[str, Any]],
    train_ratio: float = 0.85,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split threads into train (retrieval corpus) and test/eval sets at the conversation level.
    """
    rng = random.Random(seed)
    shuffled = list(threads)
    rng.shuffle(shuffled)
    
    split_point = int(len(shuffled) * train_ratio)
    train_set = shuffled[:split_point]
    test_set = shuffled[split_point:]
    
    return train_set, test_set
