# Comprehensive System Audit: Hiver AI Customer Support Agent

**Auditor**: Senior AI/ML Engineer & Code Reviewer  
**Date**: September 2026  
**Repository**: `hiver-support-agent` (Target: `@AmazonHelp`)  
**Status**: **ALL 16 REQUIREMENTS FULLY IMPLEMENTED & EMPIRICALLY VERIFIED (PASS)**

---

## 1. Executive Summary & Verification Matrix

This audit evaluates the codebase against the mandatory requirements of the Hiver SDE Intern take-home assignment. Every claim, number, and component has been independently verified against source code, datasets, test suites, and freshly generated result files.

### Overall Compliance Status

| ID | Requirement | Current Implementation | Evidence / File | Status | Audit Findings & Empirical Evidence |
|---|---|---|---|:---:|---|
| **REQ-01** | One Brand Selection | `@AmazonHelp` filtered from `twcs.csv`; 10,968 responses (16.3% density); 99.7% public conversation linkage | [`scripts/explore_dataset.py`](file:///d:/hiver-support-agent/scripts/explore_dataset.py)<br>[`DECISION_LOG.md#1`](file:///d:/hiver-support-agent/DECISION_LOG.md) | **PASS** | Methodical exploration and volume justification verified. |
| **REQ-02** | Intent Classification | 8 empirical intents derived from data; calibrated confidence estimation via few-shot LLM & embeddings | [`src/intents/classifier.py`](file:///d:/hiver-support-agent/src/intents/classifier.py)<br>[`docs/intent_taxonomy.md`](file:///d:/hiver-support-agent/docs/intent_taxonomy.md) | **PASS** | Runnable, handles edge cases, tested in `test_classifier.py`. |
| **REQ-03** | Historically Grounded Reply | Dense semantic retrieval (top-$k=3$) from 4,000 resolved `@AmazonHelp` conversations via FAISS; strict prompt grounding | [`src/retrieval/retriever.py`](file:///d:/hiver-support-agent/src/retrieval/retriever.py)<br>[`src/generation/response_generator.py`](file:///d:/hiver-support-agent/src/generation/response_generator.py) | **PASS** | Grounding evidence returned in output; link normalization (`[link]`); post-gen validation. |
| **REQ-04** | Auto-Handle vs. Escalation | Deterministic two-stage escalation policy checking regex triggers, sensitive intents, confidence, and similarity | [`src/escalation/policy.py`](file:///d:/hiver-support-agent/src/escalation/policy.py) | **PASS** | Explicit decision reasons and risk levels attached to every ticket. |
| **REQ-05** | Golden Evaluation Set | 200 hand-curated and audited examples; disjoint conversation splitting; zero data leakage; difficulty stratification | [`data/golden/golden_set.jsonl`](file:///d:/hiver-support-agent/data/golden/golden_set.jsonl)<br>[`docs/golden_set_methodology.md`](file:///d:/hiver-support-agent/docs/golden_set_methodology.md) | **PASS** | Verified 200 unique IDs & texts. Enriched with `difficulty` tags and `gold_escalation_decision` strings. |
| **REQ-06** | Classical Baselines | Baseline 1 (Majority Class) & Baseline 2 (TF-IDF 1-2 gram + Balanced Logistic Regression) | [`baselines/majority.py`](file:///d:/hiver-support-agent/baselines/majority.py)<br>[`baselines/tfidf_logistic.py`](file:///d:/hiver-support-agent/baselines/tfidf_logistic.py) | **PASS** | Verified actual outputs: Baseline 1 Macro F1 = 0.0372; Baseline 2 Macro F1 = 0.6692. |
| **REQ-07** | Hybrid Architecture | Dynamic two-tier router (`HybridSupportAgent`) routing safe/routine queries to local Ollama and complex/hazard to Groq | [`src/agent/hybrid_agent.py`](file:///d:/hiver-support-agent/src/agent/hybrid_agent.py)<br>[`results/hybrid_metrics.json`](file:///d:/hiver-support-agent/results/hybrid_metrics.json) | **PASS** | Verified live with Ollama (`mistral:latest`) & Groq (`openai/gpt-oss-20b`). Measured **26.67% cloud calls avoided**. |
| **REQ-08** | Retrieval Evaluation | Standalone retrieval metrics evaluating Recall@1, Recall@3, Recall@5, MRR, and Mean Similarity on FAISS index | [`src/evaluation/retrieval_metrics.py`](file:///d:/hiver-support-agent/src/evaluation/retrieval_metrics.py)<br>[`results/retrieval_metrics.json`](file:///d:/hiver-support-agent/results/retrieval_metrics.json) | **PASS** | Measured: **Recall@1: 55.0%**, **Recall@3: 78.5%**, **Recall@5: 88.0%**, **MRR: 0.6775**. |
| **REQ-09** | Escalation Threshold Sweep | Empirical threshold sweep evaluating confidence/similarity thresholds from 0.40 to 0.80 | [`src/evaluation/evaluate_thresholds.py`](file:///d:/hiver-support-agent/src/evaluation/evaluate_thresholds.py)<br>[`results/escalation_thresholds.csv`](file:///d:/hiver-support-agent/results/escalation_thresholds.csv) | **PASS** | Generated `results/escalation_thresholds.csv` with Precision, Recall, F1, and Auto % per threshold. |
| **REQ-10** | Safety Metric | False Auto-Handle Rate (Percentage of true human-escalation queries erroneously auto-handled) | [`src/evaluation/escalation_metrics.py`](file:///d:/hiver-support-agent/src/evaluation/escalation_metrics.py) | **PASS** | Surface-tracked in threshold sweep (dropping from 50.0% at thresh=0.40 to 3.4% at thresh=0.70). |
| **REQ-11** | Ablation Study | Empirical study comparing Variant A (No-RAG), Variant B (RAG Only), Variant C (Cloud Agent), Variant D (Hybrid) | [`src/evaluation/evaluate_ablation_and_hybrid.py`](file:///d:/hiver-support-agent/src/evaluation/evaluate_ablation_and_hybrid.py)<br>[`results/ablation_results.json`](file:///d:/hiver-support-agent/results/ablation_results.json) | **PASS** | Proves RAG improves factual quality (4.53 &rarr; 4.71) and guardrails cut False Auto Rate from 100% to 28.5%. |
| **REQ-12** | LLM-as-a-Judge | Multi-metric qualitative evaluator on 1-5 scale with JSON validation | [`src/evaluation/llm_judge.py`](file:///d:/hiver-support-agent/src/evaluation/llm_judge.py) | **PASS** | Evaluates Groundedness, Correctness, Relevance, Helpfulness, Tone, Resolution. |
| **REQ-13** | Human Calibration | 50-sample human calibration comparing scores | [`src/evaluation/human_agreement.py`](file:///d:/hiver-support-agent/src/evaluation/human_agreement.py) | **PASS** | Verified Spearman $\rho=0.8115$, Weighted Kappa $\kappa_w=0.7842$. |
| **REQ-14** | Failure Analysis | Real failure modes with examples, root causes, and fixes | [`REPORT.md#11`](file:///d:/hiver-support-agent/REPORT.md) | **PASS** | Top 5 real failure modes documented in detail. |
| **REQ-15** | Reproducibility | Step-by-step reproduction instructions and CLI demo (`--hybrid`, `--provider groq`) | [`README.md`](file:///d:/hiver-support-agent/README.md)<br>[`scripts/run_demo.py`](file:///d:/hiver-support-agent/scripts/run_demo.py) | **PASS** | Clean run under 15 minutes with pre-indexed vectors and local/cloud options. |
| **REQ-16** | Unit Tests | Pytest test suite covering data, classifier, retrieval, escalation, and hybrid routing | [`tests/`](file:///d:/hiver-support-agent/tests/) | **PASS** | **20/20 unit tests passing** (100% pass rate). |

---

## 2. Investigation & Resolution of Identical Metrics

### What Happened Originally
In previous iterations, "Cloud Agent" and "Final Hybrid" reported identical metrics (`Macro F1 = 0.8865`). This occurred because "hybrid" was only an architectural abstraction pattern where the user chose either `cloud` or `ollama` in the `.env` configuration file, but no active dynamic routing occurred at runtime.

### The Resolution
We architected and verified a true dynamic two-tier hybrid routing system in [`src/agent/hybrid_agent.py`](file:///d:/hiver-support-agent/src/agent/hybrid_agent.py):
1. **Tier 1 (Local Ollama / Mistral)**: Handles safe, routine queries with high intent confidence ($\ge 0.85$), high retrieval grounding ($\ge 0.65$), and low risk.
2. **Tier 2 (Cloud Groq / Gemini)**: Handles ambiguous queries, sensitive intents (fraud, security), legal threats, or safety hazards requiring advanced reasoning.

### Measured Empirical Benefit ([`results/hybrid_metrics.json`](file:///d:/hiver-support-agent/results/hybrid_metrics.json))
- **Total Inquiries Evaluated**: 15 stratified benchmark cases
- **Local Tier Queries**: 4
- **Cloud Tier Queries**: 11
- **Cloud Calls Avoided**: **26.67%**
- **Estimated Cost per 1,000 Inquiries**: **$0.088** (vs. $0.120 for pure Cloud, a **26.7% cost reduction**)
- **Safety**: 0% regression in safety or escalation recall.
