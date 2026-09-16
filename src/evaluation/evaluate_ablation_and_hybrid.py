"""
src/evaluation/evaluate_ablation_and_hybrid.py

Runs ablation study and hybrid architecture evaluation:
Ablation Variants:
- Variant A: Zero-Shot LLM (No-RAG, No Escalation Guardrails)
- Variant B: RAG Only (Dense Retrieval, No Escalation Guardrails)
- Variant C: Cloud Only Agent (Dense Retrieval + Deterministic Escalation + Groq)
- Variant D: Hybrid Agent (Local Ollama for safe/high-confidence + Cloud Groq for complex)

Measures:
- Intent Macro F1
- Escalation F1
- False Auto-Handle Rate (Safety failure %)
- Mean Reply Groundedness & Quality
- Cloud Calls Avoided (%)
- Latency (p50, p95 ms)
- Cost per 1,000 requests ($)

Saves results to:
- results/ablation_results.json
- results/hybrid_metrics.json
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

from src.retrieval.retriever import SupportCaseRetriever
from src.escalation.policy import EscalationPolicy
from src.escalation.decision import EscalationAction, RiskLevel
from src.generation.response_generator import ResponseGenerator
from src.intents.classifier import IntentClassifier
from src.agent.support_agent import SupportAgent
from src.agent.hybrid_agent import HybridSupportAgent
from src.llm.factory import get_llm_provider
from src.evaluation.intent_metrics import compute_intent_metrics
from src.evaluation.escalation_metrics import compute_escalation_metrics
from src.evaluation.reply_metrics import compute_reply_metrics

def run_evaluation_study(
    golden_set_path: str = "data/golden/golden_set.jsonl",
    sample_size: int = 40
) -> Dict[str, Any]:
    with open(golden_set_path, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    # Stratified subset covering diverse difficulties
    if sample_size and len(cases) > sample_size:
        # Step through evenly to preserve class and difficulty distribution
        step = len(cases) // sample_size
        test_sample = [cases[i] for i in range(0, len(cases), step)][:sample_size]
    else:
        test_sample = cases

    print(f"Running study across {len(test_sample)} stratified evaluation queries...")

    retriever = SupportCaseRetriever()
    cloud_llm = get_llm_provider(provider="groq")

    # Variant A: Zero-Shot (No-RAG, No Escalation Rules)
    print("\n[1/4] Evaluating Variant A: Zero-Shot LLM (No RAG, No Safety Guardrails)...")
    a_preds = []
    a_replies = []
    a_latencies = []
    no_rag_generator = ResponseGenerator(llm_provider=cloud_llm)
    classifier = IntentClassifier(llm_provider=cloud_llm)

    for item in test_sample:
        t0 = time.perf_counter()
        query = item["customer_msg"]
        intent_res = classifier.classify(query)
        intent = intent_res["intent"]
        # Generate with zero evidence
        reply = no_rag_generator.generate(query, intent, evidences=[])
        lat = (time.perf_counter() - t0) * 1000
        a_latencies.append(lat)
        a_preds.append(intent)
        a_replies.append(reply)

    # Variant B: RAG Only (No Escalation Guardrails)
    print("[2/4] Evaluating Variant B: RAG Only (Dense Retrieval, No Safety Guardrails)...")
    b_replies = []
    b_latencies = []
    for item in test_sample:
        t0 = time.perf_counter()
        query = item["customer_msg"]
        gold_intent = item["gold_intent"]
        evs = retriever.retrieve(query, top_k=3)
        reply = no_rag_generator.generate(query, gold_intent, evidences=evs)
        lat = (time.perf_counter() - t0) * 1000
        b_latencies.append(lat)
        b_replies.append(reply)

    # Variant C: Cloud Only Agent (RAG + Escalation Guardrails + Groq)
    print("[3/4] Evaluating Variant C: Cloud Only Agent (RAG + Guardrails + Groq)...")
    cloud_agent = SupportAgent(llm_provider=cloud_llm, retriever=retriever)
    c_results = []
    c_latencies = []
    for item in test_sample:
        t0 = time.perf_counter()
        res = cloud_agent.process(item["customer_msg"])
        lat = (time.perf_counter() - t0) * 1000
        c_latencies.append(lat)
        c_results.append(res)

    # Variant D: Full Hybrid Agent (Local Ollama for safe/high-confidence + Cloud Groq)
    print("[4/4] Evaluating Variant D: Full Hybrid Agent (Local + Cloud Routing)...")
    hybrid_agent = HybridSupportAgent(retriever=retriever)
    d_results = []
    d_latencies = []
    local_routed_count = 0
    cloud_routed_count = 0

    for item in test_sample:
        t0 = time.perf_counter()
        res = hybrid_agent.process(item["customer_msg"])
        lat = (time.perf_counter() - t0) * 1000
        d_latencies.append(lat)
        d_results.append(res)
        if "local" in res.get("routed_tier", ""):
            local_routed_count += 1
        else:
            cloud_routed_count += 1

    # Compute Metrics
    gold_intents = [item["gold_intent"] for item in test_sample]
    gold_escalations = [bool(item.get("gold_should_escalate")) for item in test_sample]
    reference_replies = [item.get("reference_reply", "") for item in test_sample]

    # Variant A Metrics
    a_intent_metrics = compute_intent_metrics(gold_intents, a_preds)
    a_reply_metrics = compute_reply_metrics(reference_replies, a_replies)
    # Variant A has no escalation (always auto-handles): 100% false auto-handle rate on true escalations
    a_false_auto_handle = 100.0

    # Variant B Metrics
    b_reply_metrics = compute_reply_metrics(reference_replies, b_replies)
    b_false_auto_handle = 100.0

    # Variant C Metrics
    c_preds = [r["intent"] for r in c_results]
    c_intent_metrics = compute_intent_metrics(gold_intents, c_preds)
    c_esc_preds = [bool(r["should_escalate"]) for r in c_results]
    c_esc_metrics = compute_escalation_metrics(gold_escalations, c_esc_preds)
    c_reply_metrics = compute_reply_metrics(reference_replies, [r["reply"] for r in c_results])

    # Variant D Metrics
    d_preds = [r["intent"] for r in d_results]
    d_intent_metrics = compute_intent_metrics(gold_intents, d_preds)
    d_esc_preds = [bool(r["should_escalate"]) for r in d_results]
    d_esc_metrics = compute_escalation_metrics(gold_escalations, d_esc_preds)
    d_reply_metrics = compute_reply_metrics(reference_replies, [r["reply"] for r in d_results])

    # Compute Cost Estimates per 1,000 queries
    # Ollama cost: $0.00
    # Groq cost: ~1,200 tokens/query @ $0.10 / 1M tokens -> ~$0.00012 / query -> $0.12 per 1,000 queries
    cost_cloud_per_1k = 0.12
    cost_ollama_per_1k = 0.00
    cloud_calls_avoided_pct = round(local_routed_count / len(test_sample) * 100, 2)
    hybrid_cost_per_1k = round((cloud_routed_count / len(test_sample)) * cost_cloud_per_1k, 4)

    ablation_summary = [
        {
            "variant": "Variant A: Zero-Shot (No RAG, No Guardrails)",
            "intent_macro_f1": a_intent_metrics["macro_f1"],
            "reply_quality": a_reply_metrics["mean_overall"],
            "escalation_f1": 0.0,
            "false_auto_handle_rate_pct": 100.0,
            "p95_latency_ms": round(float(np.percentile(a_latencies, 95)), 1),
            "cloud_usage_pct": 100.0,
            "estimated_cost_per_1k": cost_cloud_per_1k
        },
        {
            "variant": "Variant B: RAG Only (No Guardrails)",
            "intent_macro_f1": a_intent_metrics["macro_f1"],
            "reply_quality": b_reply_metrics["mean_overall"],
            "escalation_f1": 0.0,
            "false_auto_handle_rate_pct": 100.0,
            "p95_latency_ms": round(float(np.percentile(b_latencies, 95)), 1),
            "cloud_usage_pct": 100.0,
            "estimated_cost_per_1k": cost_cloud_per_1k
        },
        {
            "variant": "Variant C: Cloud Only Agent (RAG + Guardrails + Groq)",
            "intent_macro_f1": c_intent_metrics["macro_f1"],
            "reply_quality": c_reply_metrics["mean_overall"],
            "escalation_f1": c_esc_metrics["f1_score"],
            "false_auto_handle_rate_pct": round(c_esc_metrics.get("false_auto_handle_rate", 0.0) * 100, 2),
            "p95_latency_ms": round(float(np.percentile(c_latencies, 95)), 1),
            "cloud_usage_pct": 100.0,
            "estimated_cost_per_1k": cost_cloud_per_1k
        },
        {
            "variant": "Variant D: Full Hybrid Agent (Local + Cloud Routing)",
            "intent_macro_f1": d_intent_metrics["macro_f1"],
            "reply_quality": d_reply_metrics["mean_overall"],
            "escalation_f1": d_esc_metrics["f1_score"],
            "false_auto_handle_rate_pct": round(d_esc_metrics.get("false_auto_handle_rate", 0.0) * 100, 2),
            "p95_latency_ms": round(float(np.percentile(d_latencies, 95)), 1),
            "cloud_usage_pct": round(cloud_routed_count / len(test_sample) * 100, 1),
            "estimated_cost_per_1k": hybrid_cost_per_1k
        }
    ]


    hybrid_metrics = {
        "total_requests": len(test_sample),
        "local_requests": local_routed_count,
        "cloud_requests": cloud_routed_count,
        "cloud_calls_avoided_pct": cloud_calls_avoided_pct,
        "avg_latency_ms": round(float(np.mean(d_latencies)), 2),
        "p50_latency_ms": round(float(np.median(d_latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(d_latencies, 95)), 2),
        "estimated_cost_per_1k": hybrid_cost_per_1k,
        "cloud_savings_pct": cloud_calls_avoided_pct
    }

    # Save to JSON
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(ablation_summary, f, indent=2)
    with open(out_dir / "hybrid_metrics.json", "w", encoding="utf-8") as f:
        json.dump(hybrid_metrics, f, indent=2)

    return {"ablation": ablation_summary, "hybrid": hybrid_metrics}

if __name__ == "__main__":
    study = run_evaluation_study(sample_size=15)
    print("\n" + "=" * 90, flush=True)

    print("EMPIRICAL COMPONENT ABLATION STUDY RESULTS")
    print("=" * 90)
    print(f"{'Variant':<45} {'Intent F1':<12} {'Quality':<10} {'Esc F1':<10} {'False Auto %':<14} {'p95 (ms)':<10}")
    print("-" * 90)
    for v in study["ablation"]:
        print(f"{v['variant']:<45} {v['intent_macro_f1']:<12.4f} {v['reply_quality']:<10.2f} {v['escalation_f1']:<10.4f} {v['false_auto_handle_rate_pct']:<14.2f} {v['p95_latency_ms']:<10.1f}")
    print("=" * 90)
    print("\nHYBRID ROUTING BENEFITS:")
    h = study["hybrid"]
    print(f"  Total Requests:         {h['total_requests']}")
    print(f"  Local Tier (Ollama):    {h['local_requests']}")
    print(f"  Cloud Tier (Groq):      {h['cloud_requests']}")
    print(f"  Cloud Calls Avoided:    {h['cloud_calls_avoided_pct']}%")
    print(f"  Estimated Cost / 1k:    ${h['estimated_cost_per_1k']}")
    print("=" * 90)
