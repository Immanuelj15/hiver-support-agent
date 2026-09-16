"""
src/evaluation/evaluate_thresholds.py

Runs an empirical escalation threshold sweep across multiple similarity / confidence thresholds:
[0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
Measures:
- Auto-handled %
- Escalated %
- Escalation Precision, Recall, F1
- False Auto-Handle Rate (Safety failure: true escalation incorrectly auto-handled)
- False Escalation Rate (Unnecessary escalations: true auto-handle escalated)
Generates:
- results/escalation_thresholds.csv
- results/escalation_sweep_summary.json
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, List

from src.retrieval.retriever import SupportCaseRetriever
from src.escalation.policy import EscalationPolicy
from src.escalation.decision import EscalationAction

def run_threshold_sweep(
    golden_path: str = "data/golden/golden_set.jsonl",
    thresholds: List[float] = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
) -> List[Dict[str, Any]]:
    with open(golden_path, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    retriever = SupportCaseRetriever()

    # Pre-fetch retrieval similarities and mock/pre-calculated classifier confidences to ensure uniform comparison
    precomputed = []
    for item in cases:
        query = item["customer_msg"]
        gold_intent = item["gold_intent"]
        gold_escalate = bool(item.get("gold_should_escalate"))
        
        # Dense retrieval
        results = retriever.retrieve(query, top_k=1)
        sim = results[0]["similarity"] if results else 0.0

        # Confidence heuristic: exact intent match gets 0.90, related gets 0.70, mismatch gets 0.50
        conf = 0.90 if results and results[0].get("intent") == gold_intent else 0.65

        precomputed.append({
            "query": query,
            "gold_intent": gold_intent,
            "gold_escalate": gold_escalate,
            "sim": sim,
            "conf": conf
        })

    sweep_results = []

    for thresh in thresholds:
        # Instantiate policy with varying retrieval_threshold and confidence_threshold
        policy = EscalationPolicy(
            confidence_threshold=thresh,
            retrieval_threshold=thresh
        )

        tp = 0  # True Escalation (Gold: Escalate, Pred: Escalate)
        fp = 0  # False Escalation (Gold: Auto, Pred: Escalate)
        tn = 0  # True Auto-Handle (Gold: Auto, Pred: Auto)
        fn = 0  # False Auto-Handle (Gold: Escalate, Pred: Auto) -> CRITICAL SAFETY FAILURE

        for item in precomputed:
            dec = policy.evaluate(
                customer_msg=item["query"],
                predicted_intent=item["gold_intent"],
                intent_confidence=item["conf"],
                retrieval_similarity=item["sim"]
            )
            pred_escalate = (dec.action == EscalationAction.ESCALATE)
            gold_escalate = item["gold_escalate"]

            if pred_escalate and gold_escalate:
                tp += 1
            elif pred_escalate and not gold_escalate:
                fp += 1
            elif not pred_escalate and not gold_escalate:
                tn += 1
            else:
                fn += 1

        total = len(precomputed)
        auto_handled_pct = round((tn + fn) / total * 100, 2)
        escalated_pct = round((tp + fp) / total * 100, 2)

        prec = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        rec = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1 = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else 0.0

        # False Auto-Handle Rate = FN / Total Gold Escalations (missed critical tickets)
        total_gold_escalate = tp + fn
        false_auto_handle_rate = round(fn / total_gold_escalate * 100, 2) if total_gold_escalate > 0 else 0.0

        # False Escalation Rate = FP / Total Gold Auto-handles (unnecessary human toil)
        total_gold_auto = tn + fp
        false_escalation_rate = round(fp / total_gold_auto * 100, 2) if total_gold_auto > 0 else 0.0

        sweep_results.append({
            "threshold": thresh,
            "auto_handled_pct": auto_handled_pct,
            "escalated_pct": escalated_pct,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "false_auto_handle_rate": false_auto_handle_rate,
            "false_escalation_rate": false_escalation_rate,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn
        })

    # Save to CSV
    csv_path = Path("results/escalation_thresholds.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "threshold", "auto_handled_pct", "escalated_pct", "precision", 
            "recall", "f1", "false_auto_handle_rate", "false_escalation_rate"
        ], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sweep_results)

    # Save full JSON
    json_path = Path("results/escalation_sweep_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)

    return sweep_results

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ESCALATION THRESHOLD EMPIRICAL SWEEP")
    print("=" * 80)
    res = run_threshold_sweep()
    print(f"{'Thresh':<8} {'Auto %':<10} {'Esc %':<10} {'Prec':<10} {'Rec':<10} {'F1':<10} {'False Auto-Handle %':<20} {'False Esc %':<15}")
    print("-" * 80)
    for r in res:
        print(f"{r['threshold']:<8.2f} {r['auto_handled_pct']:<10.1f} {r['escalated_pct']:<10.1f} {r['precision']:<10.4f} {r['recall']:<10.4f} {r['f1']:<10.4f} {r['false_auto_handle_rate']:<20.2f} {r['false_escalation_rate']:<15.2f}")
    print("=" * 80)
    print("Saved results to results/escalation_thresholds.csv and results/escalation_sweep_summary.json")
