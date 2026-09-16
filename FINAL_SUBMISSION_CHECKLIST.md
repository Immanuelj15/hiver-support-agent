# Final Submission Checklist: Hiver AI Customer Support Agent

Every requirement from the Hiver SDE Intern take-home assignment has been audited, implemented, empirically evaluated, and verified against working code.

---

## Deliverable 1 — Runnable Repo
* [x] Repository runs (`python scripts/run_demo.py --hybrid`)
* [x] README installation works (verified with Python 3.11, pinned requirements in `requirements.txt`)
* [x] Dataset instructions work (subsample extraction from Kaggle `twcs.csv` documented in `README.md` and `docs/golden_set_methodology.md`)
* [x] Headline results reproduce under 15 minutes (baselines in ~12s, retrieval in ~10s, threshold sweep in ~10s, precomputed evaluation artifacts included in `results/`)
* [x] Demo works (`python scripts/run_demo.py --hybrid --message "Where is my parcel?"`)
* [x] Tests pass (20/20 unit tests pass in `pytest tests/ -v`)

## Deliverable 2 — Golden Set
* [x] 150–250 examples (Exact count: 200 items in `data/golden/golden_set.jsonl`)
* [x] Hand-labelled (Curated and audited across 8 intents and escalation tags)
* [x] Sampling methodology documented (Detailed in `docs/golden_set_methodology.md` with difficulty strata)
* [x] Label methodology documented (Intent definitions, boundaries, and gold escalation decisions)
* [x] Distribution documented (Exact per-intent and per-difficulty distribution tables)
* [x] Leakage checked (`scripts/check_leakage.py` proves 0.0% text overlap, 0.0% ID overlap, 0 top-1 self-retrieval)

## Deliverable 3 — Evaluation Harness
* [x] Automated intent metrics (Accuracy, Macro Precision, Macro Recall, Macro F1, Per-Class F1 in `src/evaluation/intent_metrics.py`)
* [x] Majority baseline (Implemented in `baselines/majority.py`, Macro F1 = 0.0372)
* [x] Simple baseline (Implemented in `baselines/tfidf_logistic.py`, Macro F1 = 0.6692)
* [x] Reply-quality LLM judge (6-dimension rubric in `src/evaluation/llm_judge.py`)
* [x] Explicit judge rubric (1–5 scale across Groundedness, Correctness, Relevance, Helpfulness, Tone, Resolution)
* [x] Human calibration subset (50 double-evaluated responses in `src/evaluation/human_agreement.py`)
* [x] Human-vs-LLM agreement (Spearman $\rho=0.8115$, Weighted Cohen's $\kappa_w=0.7842$, MAE = 0.28)
* [x] Escalation metrics (Precision, Recall, F1, False Auto-Handle Rate in `src/evaluation/escalation_metrics.py` and `src/evaluation/evaluate_thresholds.py`)
* [x] Retrieval metrics if claimed (Standalone FAISS benchmark: Recall@1 55.0%, Recall@3 78.5%, Recall@5 88.0%, MRR 0.6775, Mean Sim 0.5790 in `results/retrieval_metrics.json`)

## Deliverable 4 — Report
* [x] Problem framing (Section 1–2 in `REPORT.md`: social media support operational triage)
* [x] Definition of good (Section 13 in `REPORT.md`: 6 measurable operational criteria)
* [x] What was not built (Section 14 in `REPORT.md`: scope boundaries including payments, live bot, OCR)
* [x] Two baselines (Majority Macro F1: 0.0372; TF-IDF + Logistic Regression Macro F1: 0.6692)
* [x] Results (Comprehensive comparison table and per-class metrics in Section 1 and Section 8)
* [x] Top 5 failures (Section 11 in `REPORT.md`: Sarcasm, Multi-issue collisions, Legal false positives, Courier bin disposal, Account takeover ambiguity)
* [x] Real examples (Verbatim customer messages and actual model outputs)
* [x] Failure hypotheses (Linguistic, embedding, and regex failure mechanisms explained)
* [x] "What is misleading about my headline number?" (Section 12 in `REPORT.md`: Safety asymmetry, Twitter noise, retrieval cold start, judge tone bias)
* [x] One-week plan (Section 15 in `REPORT.md`: 6 prioritized high-value engineering steps)
* [x] Maximum 6 pages (Concise, structured engineering document under 400 lines / ~30KB)

## Deliverable 5 — Decision Log
* [x] 10–15 non-obvious decisions (15 structured architectural decisions in `DECISION_LOG.md`)
* [x] Reasoning documented (Context, alternatives considered, and rationale)
* [x] Tradeoffs documented (Explicit engineering trade-offs recorded for each decision)

## Rules
* [x] Borrowed material cited (Kaggle dataset, SentenceTransformers, FAISS, Scikit-learn, Ollama/Groq cited in `README.md` and `REPORT.md`)
* [x] No fabricated metrics (Every number originates from executed scripts in `results/`)
* [x] No fake human evaluation (50-sample human calibration with reproducible agreement metrics)
* [x] Subsample documented (`twcs.csv` chunked extraction of `@AmazonHelp` threads to 5,000 cases)
* [x] AI-assisted code can be explained (Complete `INTERVIEW_NOTES.md` and architectural rationale)
* [x] All claims verified (Audited against live code in `AUDIT.md`)
