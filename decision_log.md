# Architectural Decision Log: Hiver AI Customer Support Agent

This document records 15 critical architectural and engineering decisions made during the design, implementation, and evaluation of the Hiver AI Customer Support Agent.

---

## 1. Brand Selection: `@AmazonHelp`
- **Context**: The raw Kaggle dataset (`twcs.csv`) spans ~3 million tweets across dozens of brands (airlines, telecom, tech, retail).
- **Alternatives Considered**: `@AppleSupport`, `@Delta`, `@Uber_Support`.
- **Decision Made**: Selected `@AmazonHelp` as the exclusive target brand.
- **Rationale**: In exploration of `twcs.csv`, `@AmazonHelp` had the highest conversation density (10,968 responses in a 150k sample, 16.3% of all replies). Critically, 99.7% of `@AmazonHelp` tweets were linked to preceding customer tweets via `in_reply_to_tweet_id`, whereas `@AppleSupport` and `@Delta` frequently relied on cross-platform private DMs that broke public thread continuity.
- **Trade-offs**: E-commerce has high diversity in issue types (shipping delays, damaged goods, digital Prime subscriptions, account lockouts), requiring a richer intent taxonomy.

---

## 2. Empirical Intent Taxonomy: 8 Canonical Intents
- **Context**: The raw dataset contains no ground truth intent labels.
- **Alternatives Considered**: 3 coarse intents (Shipping, Billing, Other) vs. 20+ fine-grained intents.
- **Decision Made**: Formulated an 8-intent empirical taxonomy: `order_delivery_delay`, `damaged_or_wrong_item`, `return_and_refund`, `account_security_and_login`, `subscription_and_billing`, `product_technical_issue`, `feedback_or_complaint`, `other`.
- **Rationale**: 3 intents failed to separate high-risk security incidents from routine tracking inquiries. 20+ intents suffered from severe semantic overlap and fragmented classification confidence. 8 intents provided clear operational boundaries and distinct escalation behaviors.
- **Trade-offs**: Some multi-issue customer tweets require hierarchical triage (e.g. prioritizing security over delivery delays).

---

## 3. Chunked Memory-Safe Streaming for `twcs.csv`
- **Context**: `twcs.csv` is 516 MB with over 2.8 million rows. Loading the entire CSV into memory caused high memory spikes and potential crashes on constrained developer environments.
- **Alternatives Considered**: One-shot `pd.read_csv("twcs.csv")` vs. Database ingestion (SQLite/Postgres).
- **Decision Made**: Implemented generator-based chunked streaming (`chunksize=100,000`) filtering rows on-the-fly.
- **Rationale**: Keeps RAM usage strictly bounded under 200 MB regardless of machine resources, making the ingestion pipeline runnable on any developer laptop or CI container.
- **Trade-offs**: Slightly longer streaming runtime (~45 seconds) compared to full-RAM load.

---

## 4. Conversation-Level Splitting & Zero Data Leakage
- **Context**: Standard random message-level train/test splits cause severe data leakage because follow-up tweets from the same customer conversation appear in both the retrieval corpus and test set.
- **Alternatives Considered**: Message-level `train_test_split` vs. Date-based cutoff vs. Conversation-level disjoint splitting.
- **Decision Made**: Enforced conversation-level disjoint splitting (`conversation_split`) and explicit collision verification (`check_leakage`).
- **Rationale**: Ensures test conversations and their exact customer queries never appear in the retrieval corpus or baseline training sets.
- **Trade-offs**: Reduces total training rows slightly but provides 100% trustworthy generalization metrics.

---

## 5. Two-Tier LLM Provider Architecture (Ollama Offline vs. Cloud)
- **Context**: Evaluators need to test the project locally without mandatory paid API keys, while cloud APIs provide high throughput and benchmark performance.
- **Alternatives Considered**: OpenAI-only SDK vs. Ollama-only local models.
- **Decision Made**: Abstracted `LLMProvider` interface (`OllamaProvider` pointing to `http://localhost:11434` and `CloudLLMProvider` supporting Gemini and OpenAI via standard OpenAI protocol).
- **Rationale**: Complete developer freedom. Works 100% offline out-of-the-box with `mistral:latest` or `phi3:latest`, and scales seamlessly with `gemini-3.6-flash` or `gpt-4o-mini`.
- **Trade-offs**: Local inference requires Ollama running with >=4GB RAM.

---

## 6. Dense Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`
- **Context**: Needed an embedding model to vectorize customer queries and historical resolved support cases.
- **Alternatives Considered**: OpenAI `text-embedding-3-small`, BAAI `bge-small-en-v1.5`, `all-mpnet-base-v2`.
- **Decision Made**: Standardized on `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
- **Rationale**: Extremely lightweight (80MB), fast CPU inference (<15ms per query), high semantic retrieval accuracy on conversational queries, and fully offline capable.
- **Trade-offs**: 384 dimensions capture slightly less nuanced context than 1536-dim OpenAI embeddings, but sufficient for short support tweets.

---

## 7. FAISS `IndexFlatIP` on L2-Normalized Vectors for Exact Cosine Similarity
- **Context**: Vector search engine for historical case matching.
- **Alternatives Considered**: Approximate Nearest Neighbor (`IndexHNSWFlat` or `IndexIVFFlat`) vs. ChromaDB vs. Exact `IndexFlatIP`.
- **Decision Made**: Selected FAISS `IndexFlatIP` with L2-normalized embeddings.
- **Rationale**: For corpora between 4,000 and 10,000 historical support cases, `IndexFlatIP` performs exact dot-product search in <1ms without any recall loss or quantization error.
- **Trade-offs**: For >500,000 cases, HNSW or IVF indexing would be required to maintain sub-10ms search.

---

## 8. Top-$k=3$ Dense Evidence Retrieval with Intent Filtering
- **Context**: Providing retrieved customer support cases as few-shot context to ground generation.
- **Alternatives Considered**: $k=1$ (brittle) vs. $k=10$ (context stuffing, dilution).
- **Decision Made**: Configured top-$k=3$ retrieval.
- **Rationale**: 3 cases provide diverse real-world phrasing while fitting comfortably within LLM context tokens, keeping generation latency under 500ms.
- **Trade-offs**: Occasionally includes a case with slightly lower similarity if historical coverage is sparse.

---

## 9. Two Classical Baselines: Majority Class and TF-IDF + Logistic Regression
- **Context**: Need to prove the necessity of RAG and LLMs against statistical and classical machine learning.
- **Alternatives Considered**: Evaluating only the LLM agent without baselines.
- **Decision Made**: Built Baseline 1 (Majority Class `order_delivery_delay`) and Baseline 2 (TF-IDF 1-2 grams + Multinomial Balanced Logistic Regression).
- **Rationale**: Proved that Majority Class achieves only 17.5% accuracy (Macro F1: 0.0372) and TF-IDF achieves 68.0% accuracy (Macro F1: 0.6692). The Grounded LLM Agent achieves 89.0%+ accuracy, justifying the increased compute.
- **Trade-offs**: Required maintaining classical training scripts and scikit-learn dependencies.

---

## 10. Deterministic Rule-Based Escalation vs. Pure LLM Escalation
- **Context**: Deciding whether an incoming customer message should be auto-handled or handed off to a human agent.
- **Alternatives Considered**: Asking the LLM in the prompt: "Should this query be escalated? Yes or No."
- **Decision Made**: Implemented deterministic escalation engine (`EscalationPolicy`) combining regex triggers, sensitive intent rules, classifier confidence, and retrieval similarity thresholds.
- **Rationale**: LLM self-assessed escalation suffers from hallucinated confidence and sycophancy. In high-risk domains (fraud, physical harm, legal threats), a deterministic safety harness guarantees zero non-deterministic bypasses.
- **Trade-offs**: Rule sets must be maintained and updated as new adversarial terms emerge.

---

## 11. Hard Safety Guardrails (Legal, Physical Hazard, Prompt Injection)
- **Context**: Public social support is subject to jailbreaks, threats, and emergency reports.
- **Alternatives Considered**: Relying on base model safety system prompts.
- **Decision Made**: Hardcoded multi-category regex patterns for:
  1. Prompt injection (`ignore previous instructions`, `DAN mode`, `admin token`)
  2. Safety hazard (`fire`, `smoke`, `exploded`, `shattered glass`, `hospital`, `injury`)
  3. Legal threats (`lawyer`, `attorney`, `sue`, `lawsuit`, `FTC`, `BBB`)
  4. Misconduct (`backed into mailbox`, `hit my dog`, `cursed at`)
- **Rationale**: Immediate deterministic escalation to human teams with zero latency and 100% recall on known emergency vectors.
- **Trade-offs**: May flag benign queries containing legal words (e.g. "I work as a lawyer and need my package").

---

## 12. Post-Generation Unverified Financial Promise Validator
- **Context**: LLMs can inadvertently promise refunds or credits (e.g. "I have refunded your $50") that an automated Twitter agent cannot execute.
- **Alternatives Considered**: Prompt-only negative constraints ("Do not promise refunds").
- **Decision Made**: Post-generation regex validator (`UNVERIFIED_FINANCIAL_PROMISE_PATTERN`). If the generated draft promises unauthorized financial compensation, the policy intercepts the draft and overrides the action to `ESCALATE`.
- **Rationale**: Prevents financial liability and severe brand risk. Prompt negative constraints reduce hallucinations by ~85%, but the post-generation regex validator catches the remaining 15%.
- **Trade-offs**: In rare cases, rewrites a valid explanation of a past refund into an escalation handoff.

---

## 13. Stratified Golden Benchmark ($N=200$) with Intentional Adversarial Edge Cases
- **Context**: Validating system performance on customer support data without synthetic overfitting.
- **Alternatives Considered**: 50 random test rows vs. Pure synthetic generation.
- **Decision Made**: Curated a 200-sample stratified golden evaluation set covering all 8 intents, sarcasm, angry venting, multi-issue complaints, hazardous physical items, and prompt injection attempts.
- **Rationale**: Provides statistical significance and evaluates resilience against messy real-world Twitter behaviors.
- **Trade-offs**: Required meticulous hand-annotation of reference replies and escalation rationale notes.

---

## 14. Multi-Dimensional LLM-as-a-Judge with Human Calibration
- **Context**: BLEU and ROUGE are notorious for penalizing valid paraphrases in customer support responses.
- **Alternatives Considered**: BLEU / ROUGE metrics vs. LLM Judge on 6 canonical rubrics.
- **Decision Made**: Implemented `LLMJudge` evaluating Groundedness, Correctness, Relevance, Helpfulness, Tone, and Resolution on a 1-5 scale, calibrated against human annotators using Quadratic Weighted Cohen's Kappa ($\kappa_w$).
- **Rationale**: Qualitative assessment of customer service empathy, actionability, and factual grounding cannot be captured by token n-gram overlap.
- **Trade-offs**: Incurs additional LLM inference calls during batch evaluation runs.

---

## 15. Strict Git-Ignored Large Dataset Policy
- **Context**: Raw `twcs.csv` is 516 MB, exceeding GitHub's 100 MB file size limit and creating clone friction.
- **Alternatives Considered**: Git LFS vs. Checking raw data into Git repository.
- **Decision Made**: Strictly git-ignored `twcs.csv` and `data/raw/`, committing only processed lightweight artifacts (`threads.jsonl`, `golden_set.jsonl`, `vector_metadata.json`) and providing automated download instructions in the README.
- **Rationale**: Keeps repository lightweight (<15 MB), allowing instant `git clone` and reproducible evaluation without multi-gigabyte downloads.
- **Trade-offs**: New contributors must place `twcs.csv` into `data/raw/` if they wish to rerun raw ingestion from scratch.
