# Technical Interview Preparation & Defense Notes

This guide provides exhaustive technical answers for defending the Hiver AI Customer Support Agent in a live engineering interview.

---

## 1. System Architecture & High-Level Design

### Q: Walk me through the end-to-end request flow when a customer tweet arrives.
1. **Sanitization & Normalization**: The raw customer message is stripped of extraneous Twitter handles (e.g. `@AmazonHelp`), HTML entities decoded (`&amp;` &rarr; `&`), contractions expanded, and URLs tokenized to `[link]`.
2. **Intent Classification**: The normalized text is passed to `IntentClassifier`, which performs few-shot in-context classification with calibrated confidence estimation using `gemini-3.6-flash` (or local `mistral:latest`).
3. **Dense Semantic Retrieval**: The query is vectorized via `sentence-transformers/all-MiniLM-L6-v2` (384-dim, L2-normalized) and queried against a FAISS `IndexFlatIP` containing historical resolved support cases. The top $k=3$ cases with their similarity scores and conversation IDs are returned.
4. **Deterministic Escalation Policy**: The `EscalationPolicy` evaluates 5 deterministic rules:
   - Does the query trigger regexes for legal action, safety hazards, or prompt injection?
   - Is the predicted intent sensitive (`account_security_and_login`)?
   - Is intent confidence below 0.65?
   - Is retrieval similarity below 0.55?
5. **Branching Execution**:
   - **If Escalated**: The agent bypasses generative self-service and produces a deterministic empathetic specialist handoff message directing the customer to secure private channels.
   - **If Auto-Handled**: The `ResponseGenerator` produces a draft grounded in the retrieved historical evidence.
6. **Post-Generation Compliance**: The generated draft is checked against `UNVERIFIED_FINANCIAL_PROMISE_PATTERN`. If the LLM promised an unauthorized refund, the action is overridden to `ESCALATE`.
7. **Telemetry & Audit Logging**: The response is returned with full metadata (action, risk level, confidence, similarity, evidence IDs, latency).

---

## 2. Machine Learning & Baseline Comparisons

### Q: Why did you implement classical baselines before testing LLMs?
> "Evaluation and proof are more important than system complexity."
Without baselines, an 85% accuracy claim is meaningless.
- **Baseline 1 (Majority Class)**: Always predicting `order_delivery_delay` yields **17.50% accuracy** and **0.0372 Macro F1** on the stratified benchmark. This proves the class distribution alone cannot solve the problem.
- **Baseline 2 (TF-IDF + Logistic Regression)**: Achieves **68.00% accuracy** and **0.6692 Macro F1**. It performs well on distinctive keywords ("password", "broken", "refund"), but fails on sarcasm, indirect phrasing, multi-intent queries, and subtle safety threats.
- **Grounded LLM Agent**: Achieves **89.0%+ accuracy** and **0.88+ Macro F1**, demonstrating that the contextual understanding of LLMs justifies the computational overhead.

### Q: Why did you choose RAG over fine-tuning a small model (e.g. LoRA on Llama-3-8B)?
1. **Dynamic Policy Updates**: In e-commerce, return windows, courier partners, and COVID/holiday shipping advisories change weekly. With RAG, updating support policies takes 1 second (updating the vector index). Fine-tuning requires retraining, regression testing, and redeployment.
2. **Strict Verifiability & Auditability**: RAG provides exact reference IDs (`conversation_id: 46519_46517`) for every answer. Fine-tuned models hallucinate plausible-sounding but unverifiable tracking details.
3. **Data Privacy**: Customer account numbers and PII in training sets risk being memorized and leaked by fine-tuned models. RAG keeps retrieval and generation decoupled.

---

## 3. Data Engineering & Leakage Prevention

### Q: How did you ensure zero data leakage between training/retrieval and evaluation?
1. **Conversation-Level Splitting**: Never split by tweet. If Customer Tweet 1 is in retrieval, Customer Followup Tweet 2 from the same thread must NEVER be in evaluation.
2. **Deterministic String & ID Collision Detection**: `check_leakage()` hashes all evaluation queries and checks against the retrieval corpus. Exact and normalized matches are strictly filtered.
3. **Fixed Random Seed**: `random_seed: 42` across all splits, FAISS indices, and evaluations ensures 100% deterministic reproducibility.

---

## 4. Deterministic Escalation & Safety

### Q: Why did you use deterministic rules for escalation instead of asking the LLM?
1. **Zero Tolerance for Safety Failures**: In customer support, an LLM misclassifying an exploding battery as "routine hardware troubleshooting" is a catastrophic liability. Deterministic regexes for safety (`fire`, `exploded`, `bleeding`, `hospital`) guarantee 100% recall on known emergency terms with zero hallucination risk.
2. **Zero Latency**: Regex checks execute in microseconds (<0.1ms), avoiding unnecessary LLM generation calls for high-risk queries.
3. **Auditability**: When an escalation occurs, the system logs the exact triggered rule (`SAFETY_HAZARD_TRIGGER`, `SENSITIVE_INTENT_RULE`, `LOW_CONFIDENCE_THRESHOLD`), allowing customer operations teams to inspect why a ticket was routed to human queues.

---

## 5. Scaling to 1 Million Messages / Day

### Q: How would you scale this architecture to handle 1,000,000 queries per day?
1,000,000 queries/day &approx; **11.6 queries/second average**, with peak traffic &approx; **50-75 QPS**.

1. **Caching Layer (Redis Semantic Cache)**:
   - Up to 40% of customer queries during peak shopping events (Prime Day, Black Friday) are identical or near-identical tracking inquiries ("Where is my order?", "Track order TBA...").
   - A Redis cache keyed on embedding cosine similarity ($\ge 0.96$) returns pre-computed responses in **<5ms**, deflecting 35-40% of LLM traffic.
2. **Vector Store Scaling (FAISS HNSW / Milvus)**:
   - For 1M historical cases, switch from exact `IndexFlatIP` to `IndexHNSWFlat` or `IndexIVFFlat` with FP16 quantization. Memory footprint: 1M vectors $\times$ 384 dims $\times$ 2 bytes = **~768 MB RAM**, easily fitting in a single Redis/Milvus instance with sub-5ms search latency.
3. **Asynchronous Streaming Queue Architecture**:
   - Ingest customer webhooks into **Apache Kafka / AWS SQS**.
   - Decouple webhook ingestion from LLM worker pools using Celery/Ray workers, preventing traffic spikes from dropping messages.
4. **Cost & Latency Breakdown**:
   - Cloud LLM (`gemini-3.6-flash`): ~$0.0001 per query &rarr; **~$100 / day** for 1M queries.
   - Self-hosted Ollama / vLLM cluster: 4x NVIDIA L4 GPUs running quantized `mistral-7b` &rarr; ~$1.50/hr $\times$ 4 $\times$ 24 = **~$144 / day**.

---

## 6. Critical Thinking: "What is misleading about your headline number?"

### Q: Be completely honest: What is misleading about your system's headline accuracy?
> "In an interview, admitting what your metric DOES NOT capture proves senior engineering maturity."

1. **Accuracy Masks Safety Asymmetry**:
   - A standard 90% accuracy metric treats all errors equally.
   - In production, misclassifying a "late delivery" as "feedback" is a minor annoyance (cost: $0).
   - Misclassifying an "account takeover / unauthorized credit card charge" as "general inquiry" and failing to escalate is a critical security breach (cost: thousands of dollars, regulatory fines, customer churn).
   - Our system optimizes for **Missed Escalation Rate (Target: <2%)**, not raw accuracy.
2. **Benchmark Golden Set Over-represents Clarity**:
   - Even though our golden set includes sarcasm and venting, real-world Twitter data contains extreme noise: multi-language code-switching, broken English, screenshots without text, and fragmented multi-tweet threads.
   - Headline metrics on single-turn benchmark queries will degrade by 5-10% in production without visual OCR and thread-stitching models.
3. **LLM Judge Leniency**:
   - LLM-as-a-Judge evaluations tend to score fluent, polite text generously (giving 4s and 5s for tone and resolution even when an answer is slightly generic).
   - This is why we measure **Human-Judge Agreement ($\kappa_w$)** to penalize sycophantic scoring drift.
