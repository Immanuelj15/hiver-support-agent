"""
scripts/check_leakage.py

Audits data leakage between the golden evaluation benchmark and training/retrieval corpora:
1. Exact customer message matches between golden set and vector retrieval corpus.
2. Conversation ID overlap between golden set and vector retrieval corpus.
3. Self-retrieval detection (whether a query retrieves itself as top-1 evidence).
4. Prompt leakage (whether golden test queries appear in few-shot prompt examples).
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.retriever import SupportCaseRetriever

def check_leakage():
    print("=" * 60)
    print("DATA LEAKAGE AUDIT")
    print("=" * 60)

    golden_path = Path("data/golden/golden_set.jsonl")
    metadata_path = Path("data/processed/vector_metadata.json")

    if not golden_path.exists():
        print(f"Error: Golden set not found at {golden_path}")
        return

    # 1. Load Golden Set
    golden_records = []
    with open(golden_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                golden_records.append(json.loads(line.strip()))

    print(f"Loaded {len(golden_records)} golden evaluation records.")

    # 2. Load Vector Store Metadata
    vector_records = []
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            vector_records = json.load(f)
    print(f"Loaded {len(vector_records)} vector store historical records.")

    # Sets for fast comparison
    vector_texts = {r.get("customer_msg", "").strip().lower() for r in vector_records}
    vector_conv_ids = {str(r.get("conversation_id", "")).strip() for r in vector_records if r.get("conversation_id")}

    exact_text_matches = []
    conv_id_matches = []

    for item in golden_records:
        g_id = item.get("id") or item.get("conversation_id", "")
        g_conv_id = str(item.get("conversation_id", "")).strip()
        g_text = item.get("customer_msg", "").strip().lower()

        if g_text in vector_texts:
            exact_text_matches.append(g_id)
        if g_conv_id and g_conv_id in vector_conv_ids:
            conv_id_matches.append(g_conv_id)

    print(f"\n1. Verbatim Text Overlap:")
    print(f"   Matches: {len(exact_text_matches)} / {len(golden_records)} ({len(exact_text_matches)/len(golden_records):.1%})")

    print(f"\n2. Conversation ID Overlap:")
    print(f"   Matches: {len(conv_id_matches)} / {len(golden_records)} ({len(conv_id_matches)/len(golden_records):.1%})")

    # 3. Test Retriever Self-Retrieval
    print("\n3. Testing Retriever Top-1 Self-Retrieval:")
    retriever = SupportCaseRetriever()
    self_retrieval_count = 0

    for item in golden_records:
        q = item.get("customer_msg", "")
        top_evidences = retriever.retrieve(q, top_k=1)
        if top_evidences:
            top_text = top_evidences[0].get("customer_msg", "").strip().lower()
            if top_text == q.strip().lower():
                self_retrieval_count += 1

    print(f"   Top-1 Self-Retrieval Count: {self_retrieval_count} / {len(golden_records)} ({self_retrieval_count/len(golden_records):.1%})")

    # 4. Prompt / Few-Shot Leakage Check
    print("\n4. Checking Few-Shot Prompt Templates:")
    prompt_files = list(Path("src/").rglob("*.py"))
    prompt_leaks = 0
    sample_golden_texts = [item.get("customer_msg", "").strip() for item in golden_records[:10]]
    for pf in prompt_files:
        content = pf.read_text(encoding="utf-8")
        for st in sample_golden_texts:
            if len(st) > 20 and st in content:
                prompt_leaks += 1
                print(f"   Warning: Potential leak of query '{st[:30]}...' in {pf}")

    if prompt_leaks == 0:
        print("   Zero prompt template leakage found.")

    summary = {
        "golden_set_size": len(golden_records),
        "retrieval_corpus_size": len(vector_records),
        "verbatim_text_overlap": len(exact_text_matches),
        "conversation_id_overlap": len(conv_id_matches),
        "top1_self_retrieval_count": self_retrieval_count,
        "prompt_template_leakage": prompt_leaks
    }

    results_path = Path("results/leakage_audit.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAudit complete. Results saved to {results_path}")
    print("=" * 60)

if __name__ == "__main__":
    check_leakage()
