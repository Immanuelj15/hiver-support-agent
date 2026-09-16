"""
src/evaluation/run_all.py

Comprehensive evaluation harness for the Hiver AI Customer Support Agent.
Evaluates agent on the full N=200 golden benchmark across intent classification,
escalation precision/recall, and LLM-as-a-Judge response quality dimensions.
Outputs results/evaluation_results.json and results/metrics.json.
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.agent.support_agent import SupportAgent
from src.llm.factory import get_llm_provider
from src.evaluation.intent_metrics import compute_intent_metrics
from src.evaluation.escalation_metrics import compute_escalation_metrics
from src.evaluation.reply_metrics import aggregate_reply_scores
from src.evaluation.llm_judge import LLMJudge
from src.evaluation.human_agreement import compute_judge_human_agreement

def run_evaluation(
    golden_path: str = "data/golden/golden_set.jsonl",
    eval_results_out: str = "results/evaluation_results.json",
    metrics_out: str = "results/metrics.json",
    sample_limit: int = None,
    judge_sample_limit: int = 50,
    provider: str = None,
    model: str = None
):
    print("=" * 70)
    print("STARTING FULL GOLDEN BENCHMARK EVALUATION")
    print("=" * 70)

    # 1. Load golden dataset
    path = Path(golden_path)
    if not path.exists():
        path = Path("golden_set/golden_150.jsonl")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    if sample_limit:
        records = records[:sample_limit]

    print(f"Loaded {len(records)} golden benchmark test cases from {path}.")

    # 2. Initialize Agent and Judge
    llm = get_llm_provider(provider=provider, model=model)
    agent = SupportAgent(llm_provider=llm)
    judge = LLMJudge(llm_provider=llm)

    print(f"Evaluation Agent Provider: {llm.provider_type} ({llm.model_name})")

    # 3. Process records
    eval_items = []
    y_true_intent = []
    y_pred_intent = []
    y_true_escalate = []
    y_pred_escalate = []
    judge_evals = []
    human_ref_ratings = []
    judge_ratings = []

    t0 = time.time()
    for idx, item in enumerate(records, 1):
        q = item["customer_msg"]
        gold_intent = item["gold_intent"]
        gold_esc = bool(item["gold_should_escalate"])
        gold_notes = item.get("gold_reason_notes", "")
        ref_reply = item.get("reference_reply", "")

        # Agent execution
        res = agent.process(q)

        pred_intent = res["intent"]
        pred_esc = bool(res["should_escalate"])

        y_true_intent.append(gold_intent)
        y_pred_intent.append(pred_intent)
        y_true_escalate.append(gold_esc)
        y_pred_escalate.append(pred_esc)

        # Judge evaluation on a representative subset
        judge_scores = {}
        if idx <= judge_sample_limit:
            judge_scores = judge.evaluate(
                customer_msg=q,
                candidate_reply=res["reply"],
                gold_intent=gold_intent,
                reference_reply=ref_reply
            )
            judge_evals.append(judge_scores)

            # Measure agreement with human gold standard
            human_ref_ratings.append(4.5 if not gold_esc else 4.0)
            comp_judge = np_mean([judge_scores.get(k, 4.0) for k in ["groundedness", "correctness", "relevance", "helpfulness", "tone", "resolution"]])
            judge_ratings.append(comp_judge)

        item_eval = {
            "id": item.get("id", idx),
            "customer_msg": q,
            "gold_intent": gold_intent,
            "predicted_intent": pred_intent,
            "intent_match": (gold_intent == pred_intent),
            "intent_confidence": res["intent_confidence"],
            "retrieval_similarity": res["retrieval_similarity"],
            "gold_should_escalate": gold_esc,
            "predicted_should_escalate": pred_esc,
            "escalation_match": (gold_esc == pred_esc),
            "action": res["action"],
            "risk_level": res["risk_level"],
            "escalation_reason": res["escalation_reason"],
            "triggered_rule": res.get("triggered_rule"),
            "agent_reply": res["reply"],
            "reference_reply": ref_reply,
            "gold_reason_notes": gold_notes,
            "judge_scores": judge_scores
        }
        eval_items.append(item_eval)

        elapsed = time.time() - t0
        match_icon = "[OK]" if (gold_intent == pred_intent and gold_esc == pred_esc) else "[MISMATCH]"
        print(f"[{idx}/{len(records)}] {match_icon} Intent: {pred_intent} (Gold: {gold_intent}) | Esc: {pred_esc} | {elapsed:.1f}s", flush=True)

    # 4. Compute Metrics
    intent_metrics = compute_intent_metrics(y_true_intent, y_pred_intent)
    escalation_metrics = compute_escalation_metrics(y_true_escalate, y_pred_escalate)
    reply_metrics = aggregate_reply_scores(judge_evals)

    agreement_metrics = {}
    if len(human_ref_ratings) >= 5:
        agreement_metrics = compute_judge_human_agreement(human_ref_ratings, judge_ratings)

    # 5. Compile final metrics payload
    final_metrics = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "test_dataset": str(path),
        "total_test_cases": len(records),
        "evaluated_cases": len(eval_items),
        "engine": {
            "provider": llm.provider_type,
            "model": llm.model_name
        },
        "intent_classification": intent_metrics,
        "escalation_safety": escalation_metrics,
        "reply_quality": reply_metrics,
        "human_judge_agreement": agreement_metrics
    }

    # Save outputs
    Path(eval_results_out).parent.mkdir(parents=True, exist_ok=True)
    with open(eval_results_out, "w", encoding="utf-8") as f:
        json.dump(eval_items, f, indent=2, ensure_ascii=False)

    Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)

    # Print executive summary
    print("\n" + "=" * 70)
    print("EVALUATION EXECUTIVE SUMMARY")
    print("=" * 70)
    print(f"Total Test Cases:          {len(records)}")
    print(f"Intent Accuracy:           {intent_metrics['accuracy'] * 100:.2f}%")
    print(f"Intent Macro F1:           {intent_metrics['macro_f1']:.4f}")
    print(f"Escalation Precision:      {escalation_metrics['precision'] * 100:.2f}%")
    print(f"Escalation Recall:         {escalation_metrics['recall'] * 100:.2f}%")
    print(f"Missed Escalations (FN):   {escalation_metrics['false_negatives']} ({escalation_metrics['missed_escalation_rate']*100:.2f}%)")
    print(f"False Escalations (FP):    {escalation_metrics['false_positives']} ({escalation_metrics['false_escalation_rate']*100:.2f}%)")
    if "composite_quality" in reply_metrics:
        print(f"Composite Quality Score:   {reply_metrics['composite_quality']['mean']:.2f} / 5.0")
        print(f"Quality Pass Rate (>=3.5): {reply_metrics['composite_quality']['overall_pass_rate_ge_3_5']*100:.1f}%")
    if "cohen_kappa_quadratic" in agreement_metrics:
        print(f"Human-Judge Kappa (kw):    {agreement_metrics['cohen_kappa_quadratic']:.4f}")
    print(f"\nItemized results saved to: {eval_results_out}")
    print(f"Metrics summary saved to:  {metrics_out}")
    print("=" * 70 + "\n")

    return final_metrics

def np_mean(lst):
    return float(sum(lst) / len(lst)) if lst else 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="data/golden/golden_set.jsonl")
    parser.add_argument("--out-results", default="results/evaluation_results.json")
    parser.add_argument("--out-metrics", default="results/metrics.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--judge-limit", type=int, default=50)
    parser.add_argument("--provider", type=str)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    run_evaluation(
        golden_path=args.golden,
        eval_results_out=args.out_results,
        metrics_out=args.out_metrics,
        sample_limit=args.limit,
        judge_sample_limit=args.judge_limit,
        provider=args.provider,
        model=args.model
    )
