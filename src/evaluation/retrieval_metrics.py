"""
src/evaluation/retrieval_metrics.py

Evaluates the dense semantic retrieval system (FAISS IndexFlatIP) across:
- Recall@1, Recall@3, Recall@5 (whether relevant intent cases appear in top-K)
- Precision@K (fraction of top-K retrieved cases that match query intent)
- Mean Reciprocal Rank (MRR)
- Mean Cosine Similarity distribution
- Identifies retrieval failure cases (low similarity or intent mismatch)
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from src.retrieval.retriever import SupportCaseRetriever

class RetrievalEvaluator:
    def __init__(self, retriever: Optional[SupportCaseRetriever] = None):
        self.retriever = retriever or SupportCaseRetriever()

    def evaluate_dataset(
        self,
        golden_set_path: str = "data/golden/golden_set.jsonl",
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Runs retrieval evaluation across golden evaluation examples.
        """
        with open(golden_set_path, "r", encoding="utf-8") as f:
            cases = [json.loads(line) for line in f if line.strip()]

        r1_hits = 0
        r3_hits = 0
        r5_hits = 0
        reciprocal_ranks = []
        precision_at_3 = []
        top1_similarities = []
        mean_similarities = []

        retrieval_failures = []

        for item in cases:
            query = item["customer_msg"]
            gold_intent = item["gold_intent"]

            results = self.retriever.retrieve(query, top_k=top_k)
            if not results:
                reciprocal_ranks.append(0.0)
                precision_at_3.append(0.0)
                continue

            sims = [r["similarity"] for r in results]
            top1_similarities.append(sims[0])
            mean_similarities.append(float(np.mean(sims[:3])))

            # Match criteria: does retrieved historical case share the customer's intent?
            matches = [r.get("intent") == gold_intent for r in results]

            # Recall@K
            if len(matches) >= 1 and matches[0]:
                r1_hits += 1
            if any(matches[:min(3, len(matches))]):
                r3_hits += 1
            if any(matches[:min(5, len(matches))]):
                r5_hits += 1

            # Precision@3
            p3 = sum(matches[:3]) / min(3, len(matches)) if matches else 0.0
            precision_at_3.append(p3)

            # MRR
            first_rank = 0
            for idx, m in enumerate(matches, 1):
                if m:
                    first_rank = idx
                    break
            rr = 1.0 / first_rank if first_rank > 0 else 0.0
            reciprocal_ranks.append(rr)

            # Track retrieval failures: low top-1 similarity (< 0.60) or rank 1 mismatch
            if sims[0] < 0.60 or not matches[0]:
                retrieval_failures.append({
                    "id": item.get("id"),
                    "query": query,
                    "gold_intent": gold_intent,
                    "top1_similarity": round(float(sims[0]), 4),
                    "retrieved_intents": [r.get("intent") for r in results[:3]],
                    "retrieved_cases": [
                        {
                            "conversation_id": r.get("conversation_id"),
                            "snippet": r.get("customer_msg", "")[:75],
                            "similarity": round(float(r.get("similarity")), 4),
                            "intent": r.get("intent")
                        }
                        for r in results[:3]
                    ],
                    "failure_type": "low_similarity" if sims[0] < 0.60 else "intent_mismatch"
                })

        total = len(cases)
        metrics = {
            "total_queries_evaluated": total,
            "recall_at_1": round(r1_hits / total, 4),
            "recall_at_3": round(r3_hits / total, 4),
            "recall_at_5": round(r5_hits / total, 4),
            "precision_at_3": round(float(np.mean(precision_at_3)), 4),
            "mrr": round(float(np.mean(reciprocal_ranks)), 4),
            "mean_top1_similarity": round(float(np.mean(top1_similarities)), 4),
            "p50_similarity": round(float(np.median(top1_similarities)), 4),
            "p95_similarity": round(float(np.percentile(top1_similarities, 95)), 4),
            "min_similarity": round(float(np.min(top1_similarities)), 4),
            "total_retrieval_failures": len(retrieval_failures),
            "sample_failures": retrieval_failures[:5]
        }
        return metrics

if __name__ == "__main__":
    evaluator = RetrievalEvaluator()
    results = evaluator.evaluate_dataset()
    print("\n" + "=" * 60)
    print("FAISS RETRIEVAL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Evaluated:        {results['total_queries_evaluated']}")
    print(f"Recall@1:               {results['recall_at_1'] * 100:.2f}%")
    print(f"Recall@3:               {results['recall_at_3'] * 100:.2f}%")
    print(f"Recall@5:               {results['recall_at_5'] * 100:.2f}%")
    print(f"Precision@3:            {results['precision_at_3'] * 100:.2f}%")
    print(f"MRR:                    {results['mrr']:.4f}")
    print(f"Mean Top-1 Sim:         {results['mean_top1_similarity']:.4f}")
    print(f"p50 Sim:                {results['p50_similarity']:.4f}")
    print(f"p95 Sim:                {results['p95_similarity']:.4f}")
    print("=" * 60)
    
    out_file = Path("results/retrieval_metrics.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_file}")
