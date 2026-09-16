# 📦 Hiver AI Support Agent (`@AmazonHelp`)

> **An intelligent, safety-first customer service agent that drafts grounded replies using verified historical resolutions and knows exactly when to escalate to a human agent.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-100%25%20Local-purple.svg)](https://ollama.ai/)
[![Dataset](https://img.shields.io/badge/Kaggle-Customer%20Support%20on%20Twitter-orange.svg)](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🌟 What is This Project? (Plain English)

Imagine an AI customer service representative working on social media for **Amazon**.

When customers tweet complaints—ranging from late deliveries to hacked accounts—most standard AI bots either make things up (*hallucinate*) or promise refunds they cannot authorize.

**This project solves that problem through a simple principle:**
> **"A wrong auto-reply is far worse than an escalated correct one."**

Instead of guessing, this agent:
1. **Listens carefully** to understand what the customer is asking (Intent Classification).
2. **Looks up real history** to see how Amazon human agents solved the exact same problem in the past (Historical Retrieval / RAG).
3. **Checks safety rules**: If the customer mentions payment fraud, legal threats, or account security, the bot **refuses to guess and immediately hands the ticket over to a human manager** (Deterministic Escalation).

---

## 🗺️ High-Level Workflow Diagram

Here is what happens every time a customer message enters the system:

```mermaid
flowchart TD
    A([👤 Customer sends Tweet]) --> B[1. Intent Classifier\nWhat is the customer talking about?]
    
    B -->|Classifies into 8 Intents| C{Is it high risk or ambiguous?\ne.g. Hacked Account, Lawsuit, Other}
    C -->|Yes: Sensitive Intent| ESCALATE[🚨 ESCALATE TO HUMAN AGENT\nLog reason & route to specialist]
    
    C -->|No: Standard Issue| D[2. Historical Memory / RAG\nSearch 4,000 past Amazon resolutions]
    
    D --> E{Did we find a similar\nhistorical case? Sim >= 0.55}
    E -->|No: Uncharted Territory| ESCALATE
    
    E -->|Yes: Precedent Found| F[3. Grounded Reply Agent\nDraft reply matching Amazon tone\nwith verified links]
    
    F --> G{4. Safety Audit\nDoes draft promise unverified money,\nrefunds, or leak passwords?}
    G -->|Yes: Risky Claim| ESCALATE
    
    G -->|No: 100% Safe| AUTO([✅ AUTO-REPLY TO CUSTOMER\nPublish safe, grounded response])

    style A fill:#f9f9f9,stroke:#333,stroke-width:2px
    style ESCALATE fill:#ffebee,stroke:#c62828,stroke-width:2px,color:#c62828
    style AUTO fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#2e7d32
```

---

## 🔄 Step-by-Step Sequence Diagram

This sequence diagram shows the exact conversation lifecycle from input to resolution:

```mermaid
sequenceDiagram
    autonumber
    actor Customer as 👤 Customer
    participant Pipeline as ⚙️ Agent Pipeline
    participant Classifier as 🏷️ Intent Classifier
    participant Retriever as 📚 RAG Memory (4k cases)
    participant ReplyAgent as ✍️ Reply Drafter
    participant Safety as 🛡️ Escalation Guard
    actor HumanAgent as 👨‍💼 Human Support

    Customer->>Pipeline: "Where is my parcel? It was supposed to be here yesterday."
    Pipeline->>Classifier: Detect customer intent
    Classifier-->>Pipeline: Intent = order_delivery_delay (95% confidence)

    Pipeline->>Retriever: Find top-3 matching past Amazon resolutions
    Retriever-->>Pipeline: Found Case #1 (90.8% similarity, past tracking reply)

    Pipeline->>ReplyAgent: Draft response using Case #1 as factual evidence
    ReplyAgent-->>Pipeline: Draft: "I'm sorry for the delay! Check tracking here: [link]"

    Pipeline->>Safety: Evaluate safety rules (refund promises? legal threats? PII?)
    Safety-->>Pipeline: Decision = AUTO (All checks passed, high precedent)

    alt If Safe to Auto-Resolve
        Pipeline-->>Customer: ✅ Sends verified reply with tracking link
    else If Fraud, Ambiguous, or Missing Evidence
        Pipeline-->>HumanAgent: 🚨 Routes ticket to human queue with reason log
    end
```

---

## 🏛️ The 4 Pillars of the System

| Pillar | What it does | Why it protects the brand |
| :--- | :--- | :--- |
| **1. Intent Classifier** | Categorizes incoming messages into 8 concrete buckets (Delivery Delay, Damaged Goods, Returns, Account Security, Prime Billing, Tech Issues, Feedback, Other). | Distinguishes routine questions from serious account emergencies right away. |
| **2. Historical Grounding (RAG)** | Searches a curated database of 4,000 real resolved Amazon tweets to find how human agents solved similar problems. | Prevents the AI from inventing fake return policies or making empty promises. |
| **3. Safety Escalation Engine** | A strict, deterministic rule engine that audits the message and draft before anything is sent. | Catches lawsuits, chargebacks, hacked accounts, and unauthorized refund claims. |
| **4. Quality Judge & Eval Harness** | Automatically scores replies on Factuality, Tone, Helpfulness, and Safety, validated against human agreement ($\kappa_w = 0.91$). | Guarantees measurable quality with every code change. |

---

## 🚀 Quickstart: Run in < 5 Minutes

### 1. Installation & Environment Setup
```bash
git clone https://github.com/Immanuelj15/hiver-support-agent.git
cd hiver-support-agent
pip install -r requirements.txt
```

### 2. Choose Your AI Engine (Cloud or 100% Free Local)

* **Option A: Cloud API (Gemini or OpenAI)**
  ```powershell
  # Windows PowerShell
  $env:GEMINI_API_KEY="your-gemini-api-key"
  # Linux/macOS
  export GEMINI_API_KEY="your-gemini-api-key"
  ```

* **Option B: 100% Local & Free with [Ollama](https://ollama.ai/)**
  ```powershell
  # Runs Mistral or Phi-3 completely offline on your local computer
  python scripts/run_demo.py --provider ollama --message "Where is my parcel?"
  ```

---

## 💻 How to Test & Experiment

### 1. Interactive Chat Mode (Easiest for Non-Tech Users)
Type questions one by one like a real customer without restarting the script:
```bash
python scripts/run_demo.py --interactive
```
*(Or with local Ollama: `python scripts/run_demo.py --provider ollama --interactive`)*

```text
======================================================================
Hiver AI Support Agent - Interactive Session
Provider: cloud | Model: gemini-3.6-flash
Type your customer message below (or 'exit' / 'quit' to end):
======================================================================

Customer > Where is my package? It was supposed to be here yesterday.

======================================================================
AI CUSTOMER SUPPORT AGENT EXECUTION
======================================================================
Customer Inquiry:      Where is my package? It was supposed to be here yesterday.
Predicted Intent:      order_delivery_delay (Confidence: 0.98)
Retrieval Similarity:  0.9081
Pipeline Action:       AUTO_HANDLE (Risk: LOW)
Escalation Flag:       NO (Auto-handled)
Engine:                cloud (gemini-3.6-flash)
----------------------------------------------------------------------
Draft Response:
I'm so sorry about the delay with your parcel! You can track live transit
updates directly under Your Orders here: [link].
----------------------------------------------------------------------
Top Retrieved Grounding Case:
  [Ref: 46519_46517] (Sim: 0.9081)
  Customer: where is my parcel that should have been delivered yesterday?...
  Agent:    i am sorry about the delay! What information is provided on your tracking? ...
======================================================================
```

### 2. Batch Test from a File
Test 10 diverse sample inquiries at once:
```bash
python scripts/run_demo.py --file data/samples/sample_queries.txt
```

### 3. Single Query Execution (CLI)
```bash
python scripts/run_demo.py --message "Someone placed 5 orders on my account using a stolen credit card!"
```

### 3. Live 2-Tier Dynamic Hybrid Mode (Ollama + Groq Cloud)
```bash
# High-confidence routine query routes to local Ollama (100% free)
python scripts/run_demo.py --hybrid --message "Where is my parcel? It was supposed to be here yesterday."

# High-risk / legal dispute routes to Cloud LLM (Groq)
python scripts/run_demo.py --hybrid --message "I am contacting my attorney and suing Amazon for an unauthorized $500 charge!"
```

---

## 📊 Running Evaluations, Sweeps & Ablation Studies

### 1. Dedicated Retrieval Benchmark (Recall@K & MRR)
```bash
python -m src.evaluation.retrieval_metrics
```
*Evaluates dense FAISS retrieval across 200 golden queries against 4,000 resolved cases:*
- **Recall@1**: 55.00%
- **Recall@3**: 78.50%
- **Recall@5**: 88.00%
- **Precision@3**: 51.33%
- **Mean Reciprocal Rank (MRR)**: 0.6775
- **Mean Top-1 Cosine Similarity**: 0.5790
*Saves to `results/retrieval_metrics.json`.*

### 2. Empirical Escalation Threshold Sweep
```bash
python -m src.evaluation.evaluate_thresholds
```
*Sweeps confidence and similarity thresholds from 0.40 to 0.80, computing False Auto-Handle Rate and human escalation volume. Saves to `results/escalation_thresholds.csv`.*

### 3. Component Ablation Study & Hybrid Evaluation
```bash
python -m src.evaluation.evaluate_ablation_and_hybrid
```
*Evaluates all 4 system variants (Zero-Shot, RAG Only, Cloud Agent, Full Hybrid) measuring quality, safety, and cloud calls avoided. Saves to `results/ablation_results.json` and `results/hybrid_metrics.json`.*

### 4. Classical Baselines (Majority Class & TF-IDF)
```bash
# Baseline 1: Majority Class
python baselines/majority.py

# Baseline 2: TF-IDF (1-2 gram) + Balanced Logistic Regression
python baselines/tfidf_logistic.py
```
*Baseline 1 Macro F1: 0.0372; Baseline 2 Macro F1: 0.6692. Generates confusion matrix to `results/figures/confusion_matrix.png`.*

### 5. Automated Pytest Suite
```bash
pytest tests/ -v
```
*Runs all 20 unit and integration tests across intent classification, FAISS retrieval, escalation rules, no-evidence safety, and hybrid routing (20/20 passed).*

---

## 📊 Summary of Baseline & Ablation Comparison

| System Variant | Intent Macro F1 | Reply Quality (1–5) | Escalation F1 | False Auto-Handle Rate % | p95 Latency | Cloud Usage % | Est. Cost / 1k |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: Majority Class** | 0.0372 | N/A | N/A | N/A | < 1 ms | 0% | $0.00 |
| **Baseline 2: TF-IDF + Logistic Reg** | 0.6692 | N/A | N/A | N/A | ~1.2 ms | 0% | $0.00 |
| **Variant A: Zero-Shot (No RAG / Rules)** | 0.9333 | 4.53 | 0.0000 | 100.00% *(Fatal)* | 11,228 ms | 100% | $0.120 |
| **Variant B: RAG Only (No Rules)** | 0.9333 | 4.56 | 0.0000 | 100.00% *(Fatal)* | 8,149 ms | 100% | $0.120 |
| **Variant C: Cloud Agent (RAG + Rules)** | **0.9333** | **4.71** | **0.6250** | **28.57%** | **17,785 ms** | 100% | $0.120 |
| **Variant D: Full Hybrid Agent** | **0.9333** | **4.67** | **0.6250** | **28.57%** | 67,727 ms | **73.3%** | **$0.088** |

*Hybrid Routing Benefit: **26.67% of cloud calls avoided** with zero safety regression.*

---

## 🛠️ Make Commands

For convenience, standard targets are available via `make`:
- `make setup`: Install dependencies from `requirements.txt`.
- `make prepare-data`: Clean threads and build knowledge corpus with leakage checks.
- `make index`: Build the FAISS dense semantic index from the knowledge corpus.
- `make baseline`: Run Baseline 2 (TF-IDF + Logistic Regression).
- `make demo`: Launch the interactive terminal demonstration.
- `make test`: Run the full `pytest` suite.
- `make evaluate`: Run the complete evaluation benchmark.


---

## 📚 Deliverables & Documentation

- 📄 **[REPORT.md](REPORT.md)**: Exhaustive 6-page technical report covering all 13 required sections, empirical results table, failure analysis, and the critical reflection: *"What is misleading about my headline number?"*
- 📝 **[DECISION_LOG.md](DECISION_LOG.md)**: 15 structured architectural decisions detailing alternatives considered, trade-offs, and rationale.
- 🎯 **[INTERVIEW_NOTES.md](INTERVIEW_NOTES.md)**: Live interview defense guide answering tough questions on ML baselines, RAG vs fine-tuning, deterministic safety, and scaling to 1M messages/day.
- 🏷️ **[docs/intent_taxonomy.md](docs/intent_taxonomy.md)**: Formal specification of the 8 empirical intents, positive/negative examples, and escalation boundaries.
- 🔬 **[docs/golden_set_methodology.md](docs/golden_set_methodology.md)**: Quota sampling methodology, edge-case breakdown (sarcasm, multi-intent, prompt injections, hazardous materials), and annotation guidelines.

---

## 📁 Repository Structure

```text
hiver-support-agent/
├── README.md                  <- Visual guide, diagrams, quickstart & interview defense
├── REPORT.md                  <- 6-page comprehensive technical engineering report
├── DECISION_LOG.md            <- 15 architectural decisions (what, why, trade-offs)
├── INTERVIEW_NOTES.md         <- Live defense Q&A guide (architecture, ML, scale, flaws)
├── Makefile                   <- Automation commands (setup, index, demo, test, evaluate)
├── requirements.txt           <- Pinned dependencies
├── configs/
│   └── config.yaml            <- Central configuration (models, thresholds, intents)
├── docs/
│   ├── intent_taxonomy.md     <- 8 empirical intents specification & boundaries
│   └── golden_set_methodology.md <- Stratified sampling methodology (N=200)
├── data/
│   ├── raw/                   <- twcs.csv (gitignored)
│   ├── processed/             <- threads.jsonl, knowledge_corpus.jsonl, faiss_index.bin
│   ├── golden/                <- golden_set.jsonl (200 stratified benchmark items)
│   └── samples/               <- sample_queries.txt
├── src/
│   ├── data/                  <- Chunked loader, cleaner, conversation splitter
│   ├── intents/               <- Taxonomy, confidence calibration, classifier
│   ├── retrieval/             <- SentenceTransformers, FAISS vector store, retriever
│   ├── llm/                   <- Abstract LLM provider, OllamaProvider, CloudLLMProvider
│   ├── generation/            <- Grounded prompt templates, response generator
│   ├── escalation/            <- Deterministic policy, safety regexes, decision schema
│   ├── agent/                 <- Unified SupportAgent coordinator
│   └── evaluation/            <- Intent metrics, reply metrics, LLM judge, agreement
├── baselines/
│   ├── majority.py            <- Baseline 1: Majority Class ('order_delivery_delay')
│   └── tfidf_logistic.py      <- Baseline 2: TF-IDF + Balanced Logistic Regression
├── tests/
│   ├── test_classifier.py     <- Unit tests for classifier & confidence calibration
│   ├── test_retrieval.py      <- Unit tests for FAISS index & similarity search
│   ├── test_escalation.py     <- Unit tests for deterministic safety & financial regex
│   └── test_agent.py          <- Integration tests for end-to-end agent pipeline
└── results/
    ├── figures/
    │   └── confusion_matrix.png <- Confusion matrix for Baseline 2
    ├── baseline_results.json  <- Empirical baseline metrics
    ├── evaluation_results.json<- Detailed benchmark predictions & judge scores
    └── metrics.json           <- Summary evaluation metrics
```

---

## 🛡️ License
This project is open-source and available under the [MIT License](LICENSE).

