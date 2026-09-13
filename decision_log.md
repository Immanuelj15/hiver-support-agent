# Architectural & Engineering Decision Log

1. **Selected `@AmazonHelp` Over Other Brands**
   - **Decision**: Focused solely on `@AmazonHelp` for the customer support corpus.
   - **Rationale**: Amazon has the highest volume (>130k inbound interactions) and the most standardized, templated resolution procedures (tracking links, replacement flows, return lockers).
   - **Trade-off**: Forfeited multi-brand generality in favor of depth, realistic policy boundaries, and verifiable grounding.

2. **Unsupervised Data-Driven Intent Discovery (k=8) vs. Preconceived Taxonomy**
   - **Decision**: Clustered 500 representative customer messages using `sentence-transformers` and k-Means ($k=8$) to derive intent buckets.
   - **Rationale**: Real customer queries in `twcs.csv` cluster around actual operational issues (e.g. carrier scans, damaged parcels, Prime student transitions) rather than clean marketing taxonomies.
   - **Trade-off**: Discovered messier, overlapping categories (e.g., delivery delay vs. false "delivered" scans) that require explicit boundary definitions.

3. **Explicit "Other" Catch-All Intent**
   - **Decision**: Introduced an `"other"` category to absorb out-of-scope, non-English, multi-intent, or legal queries.
   - **Rationale**: Forcing every customer message into a closed taxonomy causes catastrophic misclassification on rare, high-stakes edge cases.
   - **Trade-off**: Any message classified as `"other"` automatically triggers escalation by rule, increasing human queue volume for ambiguous messages.

4. **Retrieval-Augmented Few-Shot Grounding (RAG) vs. Model Fine-Tuning**
   - **Decision**: Grounded replies using the top-$k$ historically resolved cases for the same intent rather than fine-tuning a base model.
   - **Rationale**: RAG provides transparent provenance (we can audit exactly which historical case justified each reply), prevents stale knowledge, and eliminates expensive retraining when policies change.
   - **Trade-off**: Sacrificed subtle stylistic nuances that fine-tuning could memorize, and added ~30ms of embedding search latency.

5. **Decoupled Deterministic Escalation vs. Single-Call LLM "Vibes"**
   - **Decision**: Separated the escalation routing decision into an independent deterministic policy engine rather than asking the generation LLM "should this escalate?".
   - **Rationale**: Generative models exhibit optimism bias and hallucinate compliance when asked to assess their own certainty; rule-governed gating guarantees 100% escalation on legal threats, security incidents, and missing evidence.
   - **Trade-off**: Adds a post-generation verification step and requires maintaining a curated set of high-risk keyword triggers and action patterns.

6. **Prioritized Escalation Recall Over Escalation Precision**
   - **Decision**: Optimized the escalation threshold (default 0.55) to maximize Recall on the `"escalate"` class.
   - **Rationale**: In enterprise customer support, False Negatives (auto-handling an account takeover or safety hazard) cause severe reputational and legal harm, whereas False Positives only add marginal agent review time.
   - **Trade-off**: Escalation precision drops slightly, routing some borderline queries to humans that could theoretically have been answered by a bot.

7. **Intent-Partitioned Semantic Index vs. Monolithic Flat Search**
   - **Decision**: Partitioned the 4,000-thread historical retrieval index by predicted intent bucket before computing cosine similarities.
   - **Rationale**: Prevents cross-intent semantic confusion (e.g., retrieving a "return refund" resolution for an "order delivery delay" query simply because both mention "order").
   - **Trade-off**: If the upstream intent classifier misclassifies a query, the retriever searches the wrong bucket; mitigated by fallback to global search if bucket density is low.

8. **Quadratic Weighted Kappa ($\kappa_w$) Over Simple Percentage Agreement**
   - **Decision**: Validated the LLM judge against human annotations using Cohen's Quadratic Weighted Kappa on 1-5 ordinal scales.
   - **Rationale**: Percentage agreement treats a 1-point difference (e.g. 4 vs 5) as the same penalty as a 4-point catastrophic failure (e.g. 1 vs 5); quadratic weighting accurately reflects ordinal distance.
   - **Trade-off**: Requires ordinal integer formatting and slightly more complex statistical tooling than raw accuracy.

9. **Deterministic Action Pattern Matching in Reply Verification**
   - **Decision**: Implemented regex guards against unverifiable commitments (e.g., *"I have refunded $X"*, *"credited your account"*, *"send your password"*).
   - **Rationale**: An AI agent without live ERP/CRM write-APIs must never promise that financial or backend transactions have occurred.
   - **Trade-off**: Overly cautious regexes can occasionally flag empathetic paraphrases (e.g. *"we have processed your feedback"*), requiring careful pattern tuning.

10. **Offline Compressed Embedding Caching (`embeddings_cache.npz`)**
    - **Decision**: Pre-computed and cached sentence embeddings for 4,000 historical threads to a local compressed numpy archive.
    - **Rationale**: Reduces pipeline initialization from ~40 seconds down to <150 milliseconds on local cold starts.
    - **Trade-off**: Requires invalidating the `.npz` file whenever the historical thread corpus is updated or re-ingested.

11. **Routing Non-English Tweets Directly to Escalation / Specialized Queues**
    - **Decision**: Explicitly routed Spanish, French, and non-ASCII tweets into the `"other"` / escalation flow rather than auto-translating.
    - **Rationale**: Amazon operates regional customer care teams (`@AmazonHelpFR`, `@AmazonHelpES`); English-language social agents should not provide unofficial machine translations for complex regional accounts.
    - **Trade-off**: Limits automated handling strictly to the dominant English dialect present in the dataset.

12. **Stratified Golden Set with Adversarial Edge Cases vs. Uniform Random Sample**
    - **Decision**: Hand-annotated 150 stratified examples balanced across all 8 intents, deliberately including sarcasm, multi-intent queries, empty venting, and hazardous products.
    - **Rationale**: Uniform sampling yields ~60% mundane tracking inquiries, generating deceptively high benchmark scores that collapse under real-world adversarial traffic.
    - **Trade-off**: The golden set distribution does not mirror the raw production frequency, requiring careful communication in the report.
