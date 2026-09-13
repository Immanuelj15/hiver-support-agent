# Golden Set Sampling Notes (golden_150.jsonl)

## 1. Candidate Population & Volume
- **Raw Dataset**: Kaggle *Customer Support on Twitter* (`twcs.csv`), comprising ~3,002,524 total tweets.
- **Brand Subsample**: Reconstructed 13,589 clean conversational threads specifically authored by `@AmazonHelp` and linked to inbound customer inquiries.
- **Evaluation Subset**: 150 hand-curated and rigorously audited customer interactions (`golden_150.jsonl`).

## 2. Stratification Strategy
A uniform random sample of customer support tweets on Twitter is heavily biased toward trivial tracking queries (often >55–65% "Where is my package?"). Evaluating on a naive random sample produces inflated headline accuracy numbers while completely failing to validate catastrophic edge cases (account takeover, fraud, chemical/fire hazards, regulatory threats).

To overcome this, we enforced **stratified sampling across all 8 discovered intent buckets**, allocating quotas to ensure sufficient statistical representation for low-frequency, high-consequence intents:

| Intent Class | Count | % of Golden Set | Focus & Edge Cases Included |
| :--- | :---: | :---: | :--- |
| `order_delivery_delay` | 24 | 16.0% | Sarcasm, missed guaranteed delivery, false "delivered" scans, urgent medical shipments, porching piracy. |
| `damaged_or_wrong_item` | 20 | 13.3% | Glass/shattered items, infant safety seals, high-value smartphone theft, hazardous glass returns. |
| `return_and_refund` | 22 | 14.7% | Printerless QR returns, canceled order pending holds, expired return windows, empty return allegations. |
| `account_security_and_login` | 18 | 12.0% | Brute force OTP spam, foreign fraudulent orders, compromised primary email, passkey hijacking. *(100% Escalate)* |
| `subscription_and_billing` | 20 | 13.3% | Unknown Prime renewal charges, student graduation pricing, duplicate card billing, split payments. |
| `product_technical_issue` | 18 | 12.0% | Fire TV boot loops, Kindle app crashes, Echo red ring / Wi-Fi drops, Prime Video Error 7031. |
| `feedback_or_complaint` | 16 | 10.7% | Driver property damage, rude phone agents, excess packaging waste, non-actionable customer venting. |
| `other` | 12 | 8.0% | Legal / lawsuit threats, non-English (French/Spanish), job recruiting, AWS cloud questions. *(100% Escalate)* |
| **Total** | **150** | **100.0%** | **Comprehensive benchmark** |

## 3. Exclusion Rules & Rationale
1. **Sub-4 Word Messages ("noise")**: Excluded messages such as "help", "hello??", "DM me", or bare punctuation. These contain zero actionable semantic signal and disproportionately distort intent classification.
2. **Retweets & Quoted Noise**: Excluded standard retweets or public social sharing without an explicit inquiry addressed to customer care.
3. **Tracking Links without Context**: Excluded tweets containing only an unannotated tracking URL or screenshot link without explanatory text.
4. **Non-English in Standard Queues**: Filtered out foreign language tweets from the primary retrieval database, but explicitly retained a calibrated sample in `other` to test the escalation engine's ability to divert non-English traffic to localized queues (`Amazon.fr`, `Amazon.es`).

## 4. Escalation Distribution & Positive Class Priority
- **Total Should Escalate**: 69 / 150 (46.0%)
- **Total Auto-Handle**: 81 / 150 (54.0%)

In production customer support, **Escalation is treated as the positive class**. False Negatives (failing to escalate a compromised account or legal action) carry vastly higher real-world cost than False Positives (unnecessarily routing an easy inquiry to a human). Hence, evaluation strictly prioritizes **Escalation Recall** alongside Precision.
