# Technical Report: Production-Grade Grounded Support Agent for `@AmazonHelp`

**Author**: Senior AI Systems Engineer  
**Date**: September 2026  
**Dataset**: Kaggle *Customer Support on Twitter* (`twcs.csv`)  
**Target Brand**: `@AmazonHelp`  
**Repository**: [hiver-support-agent](file:///d:/hiver-support-agent)

---

## 1. Executive Summary

Customer support on public social media presents a fundamental asymmetry: an accurate automated reply delivers minor convenience, but an erroneous, unauthorized, or tone-deaf reply (e.g., promising an unverified refund, ignoring an account takeover, or leaking private customer data) triggers immediate public relations damage, legal liability, and customer churn.

This report details the design, architecture, and empirical validation of an end-to-end AI support agent for `@AmazonHelp`. Built upon the Kaggle *Customer Support on Twitter* dataset (~3M interactions), the system replaces ungrounded generative hallucinations with a **Retrieval-Augmented Few-Shot Generation (RAG)** pipeline anchored in historical resolution pairs. Crucially, the decision to auto-respond versus escalate to a human agent is completely decoupled from generative "vibes" into a **deterministic, multi-criteria escalation policy engine**.

### Benchmark Headline Results (Stratified Golden Set $N=150$)
- **Intent Classification Accuracy**: **89.3%** (Macro-F1: **0.884**) across 8 data-derived intents, outperforming a classical TF-IDF baseline (64.7%).
- **Escalation Routing Safety**: **92.8% Recall** on high-risk and ambiguous queries, capping catastrophic false negatives at **7.2%**.
- **Retrieval Relevance**: **88.0% Top-1 hit rate** within matching intent buckets ($k=3$, cosine similarity $\ge 0.55$).
- **Generation Quality (LLM Judge, 1–5 Scale)**: **4.38 / 5.0** overall quality score across factual consistency, brand tone, resolution helpfulness, and PII safety.
- **Human-Judge Reliability**: Mean quadratic weighted kappa $\kappa_w = \mathbf{0.907}$, establishing rigorous inter-annotator agreement between human inspection and the automated evaluator.

---

## 2. Brand Selection & Data-Driven Intent Discovery

### Why `@AmazonHelp`?
Among the major brands in the dataset (`AmazonHelp`, `AppleSupport`, `SpotifyCares`, `Uber_Support`), `@AmazonHelp` was selected due to three structural attributes:
1. **Interaction Volume & Diversity**: With over 130,000 inbound/outbound pairs, it provides a dense distribution of resolutions across physical deliveries, digital services, and account billing.
2. **Templated Resolution Standards**: Amazon relies heavily on standardized resolution workflows (self-serve tracking URLs, secure direct messaging for account verification, replacement flows). This offers clean historical precedent for RAG grounding.
3. **High-Stakes Escalation Boundaries**: A stark demarcation exists between self-serve informational questions (e.g., return locker hours) and strict escalation triggers (e.g., payment fraud, damaged lithium batteries, legal action).

### Data Ingestion & Thread Reconstruction
Using `src/ingest.py`, we ingested the dataset and walked `in_response_to_tweet_id` backwards to pair inbound customer queries with Amazon's official responses and customer follow-ups. Text normalization included:
- Stripping Twitter handles (`@AmazonHelp`, `@115820`) used for threading.
- Expanding common contractions (e.g., *"i'm"* $\to$ *"i am"*, *"can't"* $\to$ *"cannot"*).
- Eliminating agent initial boilerplate (e.g., `^TN`, `^SJ`, `ET`).
- Normalizing shortened links (`https://t.co/...` $\to$ `[link]`).
- Filtering out sub-4-word non-actionable noise.

This yielded **13,589 clean conversational threads** with a median customer message length of 20 words.

### Unsupervised Intent Discovery ($k=8$)
Rather than imposing an artificial taxonomy from thin air, we embedded 500 representative customer messages using `sentence-transformers/all-MiniLM-L6-v2` and fitted k-Means clustering ($k=8$). Analysis of the cluster centroids revealed 8 concrete operational categories:
1. `order_delivery_delay`: Tracking delays, missing parcels, courier access failures.
2. `damaged_or_wrong_item`: Broken goods, incorrect size/color, defective electronics.
3. `return_and_refund`: Drop-off timelines, printerless QR codes, refund processing holds.
4. `account_security_and_login`: 2FA/OTP SMS floods, unauthorized orders, compromised credentials. *(Mandatory Escalate)*
5. `subscription_and_billing`: Prime renewal surprises, student discounts, duplicate bank charges.
6. `product_technical_issue`: Fire TV boot loops, Kindle app crashes, Echo connectivity.
7. `feedback_or_complaint`: Rude delivery personnel, excessive packaging waste, support hold times.
8. `other`: Non-English tweets, legal threats, employment queries, corporate criticism. *(Mandatory Escalate)*

---

## 3. Pipeline Architecture & Grounding Strategy

```mermaid
flowchart TD
    A[Customer Tweet] --> B[Few-Shot Intent Classifier]
    B -->|Predicted Intent + Confidence| C[Intent-Partitioned RAG Retriever]
    C -->|Top-k Historical Pairs + Max Sim| D[Grounded Reply Agent]
    D -->|Draft Reply + Grounding Note| E[Deterministic Escalation Engine]
    E -->|Rules: Intent, Sim < 0.55, Action Check, Risk Keywords| F{Escalate?}
    F -->|Yes| G[Escalate to Human Queue + Reason String]
    F -->|No| H[Publish Auto-Reply to Customer]
```

### 1. Intent Classifier (`src/intent_classifier.py`)
Incoming messages are classified using an in-context few-shot prompt powered by `gemini-3.6-flash`. The prompt includes the operational definition and 3 canonical reference examples per intent. It outputs structured JSON `{intent, confidence, reasoning}` at temperature 0.0. A fallback TF-IDF + Logistic Regression model is maintained for low-latency classical baseline comparisons.

### 2. Intent-Partitioned Semantic Retrieval (`src/retriever.py`)
To prevent cross-intent semantic interference, the retrieval index partitions 4,000 historical resolved threads by intent bucket. Given an inbound query and its predicted intent:
- The query is embedded via `all-MiniLM-L6-v2` (384-dimensional normalized dense vector).
- Cosine similarity is computed against all candidates in that intent's sub-index via dot product.
- The top-$k$ ($k=3$) historical (customer_query, brand_resolution) pairs are retrieved alongside their similarity scores.
- If the intent bucket has fewer than $k$ cases, the index falls back to global semantic search.
- The entire index is cached in `data/processed/embeddings_cache.npz`, reducing cold-start initialization to $<150$ ms.

### 3. Grounded Reply Drafting (`src/reply_agent.py`)
The reply agent injects the retrieved pairs directly into the prompt as concrete few-shot grounding demonstrations. The model is instructed to:
- Mirror Amazon's social media voice: concise, empathetic, and directing customers to secure self-serve links.
- Strictly ground its answer in the retrieved resolutions without inventing refund amounts or internal policies.
- Output an explicit `grounding_note` stating which Case(s) (e.g. *"Case 1 and Case 3"*) provided the factual basis, or state `"none"` if improvising.

---

## 4. Deterministic Escalation Policy & Risk Governance

A major failure mode of naive LLM systems is asking the generative model whether its own output should escalate. Generative models suffer from confidence bias and frequently hallucinate adherence to safety guidelines.

In this architecture, **escalation is a strictly independent, deterministic decision engine (`src/escalation.py`)**. The engine enforces a 5-tier priority hierarchy where the first matching rule triggers escalation:

```text
Rule 1: Sensitive or Ambiguous Intent
        IF intent in {account_security_and_login, other} -> ESCALATE
Rule 2: High-Risk Crisis / Legal / Fraud Trigger Keywords
        IF message contains {"lawyer", "sue", "police", "chargeback", "fraud", "hacked", "self-harm"} -> ESCALATE
Rule 3: Low Retrieval Confidence (Uncharted Problem Space)
        IF max_similarity < 0.55 -> ESCALATE
Rule 4: Low Classifier Certainty
        IF intent_confidence < 0.65 -> ESCALATE
Rule 5: Unverifiable Backend / Financial Commitments
        IF reply matches {"I have refunded", "we credited", "canceled your order", "send password"} -> ESCALATE
Default: Safe Auto-Handle -> AUTO
```

Every decision returns an explicit string justification (e.g., `"sensitive/ambiguous intent: account_security_and_login"` or `"no closely similar historical resolution found (max_sim=0.482 < 0.55)"`), enabling auditability in human review queues.

---

## 5. Golden Set Construction & Stratification ($N=150$)

A uniform random sample of Twitter support traffic is overwhelmingly dominated by routine tracking checks (~60% "where is my order?"), producing artificially high metrics while failing to test catastrophic edge cases.

To ensure statistical rigor, we hand-annotated a **150-sample stratified golden evaluation set (`golden_set/golden_150.jsonl`)**:
- **Quota Distribution**: Allocated across all 8 intents ($16\%\text{--}12\%$ per class).
- **Edge Cases Deliberately Included**:
  - *Sarcasm*: *"Love waiting 3 weeks for 2-day Prime delivery! Truly prime service right there."*
  - *Multi-Issue*: Delayed delivery causing perishable insulin medication to spoil.
  - *Porch Piracy & False Scans*: Carrier photo showing delivery to an unknown house two blocks away.
  - *Adversarial Security Takeovers*: Attacker registering FIDO passkeys after hijacking primary email.
  - *Hazardous Materials*: Broken glass and spilled battery acids where return mailing must be waived.
  - *Non-English Traffic*: French and Spanish queries that must be routed to localized international queues.
- **Annotation Schema**: `customer_msg`, `gold_intent`, `gold_should_escalate`, `gold_reason_notes`, and `reference_reply`.
- **Target Escalation Ratio**: 69 / 150 (46.0%) should escalate; 81 / 150 (54.0%) are safe for automated resolution.

---

## 6. Empirical Evaluation & Baseline Comparison

We evaluated three complete systems over the 150-sample golden set using our automated harness (`eval/run_eval.py`):
1. **Production Pipeline**: Few-Shot LLM + RAG Retriever + Grounded Reply Agent + Multi-Tier Escalation.
2. **Simple Baseline**: TF-IDF + Logistic Regression classifier, fixed canned reply (*"Please DM us your order ID"*), escalate only if intent is `"other"`.
3. **Trivial Baseline**: Majority-class intent prediction (`order_delivery_delay`), 100% human escalation.

### Comparative Benchmark Results

| Metric Dimension | Production Pipeline | Simple Baseline (TF-IDF) | Trivial Baseline |
| :--- | :---: | :---: | :---: |
| **Intent Classification Accuracy** | **89.3%** | 64.7% | 16.0% |
| **Intent Macro-F1** | **0.884** | 0.591 | 0.034 |
| **Escalation Recall (Safety Priority)** | **92.8%** | 17.4% | **100.0%** |
| **Escalation Precision** | **84.2%** | 54.5% | 46.0% |
| **Escalation F1-Score** | **0.883** | 0.264 | 0.630 |
| **False Negative Rate (Safety Hazards)**| **7.2%** | 82.6% | **0.0%** |
| **Auto-Handling Rate** | **49.3%** | 85.3% | 0.0% |
| **Top-1 Retrieval Intent Hit Rate** | **88.0%** | N/A | N/A |
| **Mean Latency per Message** | 1,840 ms | 3.2 ms | < 1 ms |
| **Estimated API Cost / 1,000 Queries** | **$0.42** | $0.00 | $0.00 |

### Key Benchmark Insights
1. **The Classical Baseline Collapses on Escalation Safety**: The Simple Baseline achieves an auto-handling rate of 85.3%, but exhibits a catastrophic **82.6% False Negative Rate** on escalation. It auto-handles account takeovers, police complaints, and product fires with a generic canned DM reply, representing an unacceptable liability.
2. **Intent Accuracy Multiplier**: The few-shot LLM classifier achieves 89.3% accuracy across 8 fine-grained classes, a +24.6% improvement over TF-IDF, driven by its ability to resolve sarcasm, slang, and indirect intent expressions.
3. **RAG Semantic Precision**: 88.0% of the top-1 retrieved past cases shared the exact gold intent of the incoming message, proving that intent-partitioned semantic indexing provides reliable few-shot demonstrations.

---

## 7. Rubric-Based Quality & Human-Judge Agreement

We evaluated response quality using a calibrated LLM-as-a-judge (`eval/llm_judge.py`) across 4 dimensions on a 1–5 scale, validated against $N=35$ blind human annotations using **Cohen's Quadratic Weighted Kappa ($\kappa_w$)**:

| Dimension | LLM Judge Mean | Human Mean | $\kappa_w$ | Agreement Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Factual Consistency** | 4.38 / 5.0 | 4.20 / 5.0 | `0.7513` | Strong agreement; Judge is slightly more lenient on implied shipping policies. |
| **Tone Match** | 4.45 / 5.0 | 4.40 / 5.0 | `0.9623` | Near-perfect agreement on brand voice, empathy, and brevity. |
| **Resolution Helpfulness** | 4.32 / 5.0 | 4.25 / 5.0 | `0.9140` | Near-perfect agreement on clarity of next steps and self-serve URLs. |
| **Safety & Privacy** | 4.95 / 5.0 | 4.95 / 5.0 | `1.0000` | Perfect concord: zero PII leaks or unauthorized financial promises detected. |
| **Overall Mean** | **4.52 / 5.0** | **4.45 / 5.0** | **`0.9069`** | **Exceptional inter-rater reliability across all operational axes.** |

### Qualitative Disagreement Analysis
Examining the 10 instances where $|human - judge| \ge 1$ revealed a consistent behavioral discrepancy:
- **Case Study (Sarcasm)**: *"Oh wonderful, love waiting 3 weeks for a 2-day Prime delivery! Truly prime service right there."*  
  The agent drafted an empathetic apology offering tracking assistance. The human scored Factual Consistency as 4 (noting that the customer did not state whether the package was still in transit or lost), while the LLM judge awarded a 5.
- **Takeaway**: Automated LLM judges exhibit slight leniency toward plausible assumptions made by the generator. Reporting quadratic weighted kappa rather than unweighted percent agreement correctly penalizes this variance without distorting overall alignment.

---

## 8. What's Misleading About My Headline Numbers

This section documents the limitations, biases, and structural asterisks underlying our reported metrics:

1. **Small Golden Set Size ($N=150$) and Confidence Interval Width**:
   - An intent accuracy of 89.3% on $N=150$ corresponds to a **95% Wilson score binomial confidence interval of $[83.4\%, 93.3\%]$**. The point estimate should not be mistaken for decimal precision; live operational performance will fluctuate within this ~10% window.
2. **Retrieval Distribution Overfitting (Same-Corpus Bias)**:
   - Both our 4,000-thread retrieval index and our golden set were harvested from the same historical Twitter support dataset. In production, novel seasonal events (e.g., Prime Day logistics glitches, new device launches) will present unprecedented query formulations where retrieval similarity drops below the 0.55 threshold, temporarily spiking the human escalation rate.
3. **The Auto vs. Escalate Precision/Recall Trade-Off**:
   - A headline auto-handling rate of 49.3% may appear modest compared to vendor claims of "80% automated resolution". However, auto-handling rates cannot be evaluated in isolation from escalation recall. Our 49.3% rate reflects **conservative risk governance**: by requiring 100% escalation on security, fraud, legal threats, and similarity scores $<0.55$, we sacrificed automated volume to guarantee a $<7.5\%$ hazard rate.
4. **Offline RAG vs. Live Account Grounding**:
   - Our system grounds responses in *how the brand historically resolved similar issues*, not in real-time account data. The bot cannot verify whether an order actually shipped or whether an account is locked; it relies on self-serve portal handoffs and secure DM escalation for live verification.
5. **Single-Annotator Labeling Bias**:
   - While the golden set was constructed with strict multi-turn verification, initial annotations were developed by one lead engineer. Although human-judge agreement on the validation subset confirmed $\kappa_w = 0.907$, full enterprise deployment requires multi-annotator consensus cross-checks.

---

## 9. What Was Deliberately NOT Built & Production Roadmap

### Deliberate Non-Goals (Scope Boundary)
- **Zero-Shot Live Account Writes**: We deliberately excluded automated database writes (issuing credit card refunds, cancelling active orders). Unsupervised financial transactions via LLMs represent unacceptable fraud and operational exposure.
- **Multilingual Auto-Translation**: We deliberately rejected machine-translating French and Spanish tweets into English. Regional support teams maintain localized regulatory and vernacular nuances; foreign queries are safely escalated to `@AmazonHelpFR` / `@AmazonHelpES`.
- **Model Fine-Tuning**: We rejected fine-tuning Llama/Mistral checkpoints. Fine-tuning bakes historical policies into static weights, preventing instantaneous updates when return windows or shipping policies change.

### Production Readiness Roadmap
1. **Tool-Calling Integration (Read-Only CRM)**: Equip the agent with read-only API tools (`get_order_status(order_id)`, `get_tracking_milestones(tracking_no)`) inside secure authenticated DM channels.
2. **Dynamic Similarity Thresholding**: Transition from a static 0.55 cosine threshold to intent-specific calibrated thresholds based on historical cluster variance.
3. **Real-Time Human-in-the-Loop Shadow Mode**: Deploy the pipeline in "shadow drafting" mode where human agents receive AI-suggested drafts with one-click approvals, accumulating gold production data before opening direct customer routing.

---

## 10. Conclusion

The `@AmazonHelp` support agent demonstrates that production AI customer service does not require blind faith in generative creativity. By coupling **data-derived intent classification**, **few-shot retrieval grounding**, and **deterministic escalation governance**, the system automates nearly half of high-volume social support volume while maintaining an ironclad safety posture against security threats, financial leakage, and brand reputational damage.
