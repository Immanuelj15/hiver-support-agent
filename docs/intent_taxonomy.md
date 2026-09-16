# Intent Taxonomy: Amazon Customer Support

This document defines the 8 empirical customer-support intents developed from iterative clustering and domain analysis of `@AmazonHelp` conversations in the Kaggle Customer Support on Twitter (`twcs.csv`) dataset.

---

## Intent Summary Matrix

| Intent Key | Description | Escalation Default | Risk Level |
|---|---|---|---|
| `order_delivery_delay` | Package late, tracking frozen, false delivery scan | Automated tracking / Self-serve | Low |
| `damaged_or_wrong_item` | Physical defect, broken seals, incorrect product received | Automated guidance / Escalate if dangerous | Medium |
| `return_and_refund` | Return label generation, drop-off questions, refund status | Automated policy / Escalate on disputed funds | Medium |
| `account_security_and_login`| Compromised account, OTP failures, 2FA lockouts, phishing | **Always Escalate** | **High / Critical** |
| `subscription_and_billing` | Unauthorized Prime charges, recurring billing, double charges | Automated status / Escalate on refund disputes | Medium-High |
| `product_technical_issue` | Kindle, Fire TV, Echo/Alexa software or hardware glitch | Automated troubleshooting / Guide | Low |
| `feedback_or_complaint` | Driver behavior, packaging complaints, policy dissatisfaction | Empathetic acknowledgment / Log feedback | Low |
| `other` | Chitchat, edge cases, cross-brand queries, ambiguous queries | Fallback / Escalate if unresolved | Variable |

---

## Detailed Intent Specifications

### 1. `order_delivery_delay`
- **Definition**: Inquiries regarding packages that have not arrived by the expected delivery date, tracking numbers with no movement or frozen statuses, or delivery scans indicating delivery when the customer did not receive the package.
- **Positive Examples**:
  - *"My order was supposed to arrive yesterday by 8pm but tracking says it has not shipped yet."*
  - *"Carrier marked package as delivered at 3 PM but my porch is completely empty."*
  - *"Where is my parcel? It was supposed to be here yesterday."*
- **Negative / Boundary Examples**:
  - *"I want a refund for the guaranteed shipping fee because it was late."* &rarr; Classified as `order_delivery_delay` or `return_and_refund`, but triggers financial escalation.
  - *"The box arrived yesterday but it was soaking wet and crushed."* &rarr; `damaged_or_wrong_item`.
- **Expected Action**:
  - Check latest carrier transit window (allow 24-48h buffer for premature carrier scans).
  - Provide direct self-serve tracking link to Amazon orders.
  - Advise checking surrounding safe locations and neighbors.

---

### 2. `damaged_or_wrong_item`
- **Definition**: Issues where the delivered package contains an incorrect item, wrong size/color, missing parts, or physically broken/shattered contents.
- **Positive Examples**:
  - *"I ordered wireless headphones and received a box containing a pair of socks."*
  - *"The glass blender jar arrived completely shattered inside the shipping box."*
  - *"Package seal was torn open and the internal electronic unit is missing."*
- **Negative / Boundary Examples**:
  - *"I ordered this by mistake and want to send it back."* &rarr; `return_and_refund`.
  - *"The device turns on but won't connect to my home Wi-Fi."* &rarr; `product_technical_issue`.
- **Expected Action**:
  - Express sincere empathy for the damaged/incorrect product.
  - Direct user to *Your Orders* &rarr; *Return or Replace Items*.
  - **Safety Policy**: If hazardous materials (shattered glass, leaking chemicals, lithium battery bulge) are detected, escalate immediately and instruct the customer not to handle broken pieces.

---

### 3. `return_and_refund`
- **Definition**: Inquiries about returning an eligible item, generating return shipping QR codes/labels, drop-off location assistance (Kohl's, Whole Foods, UPS), or checking the status of an expected refund.
- **Positive Examples**:
  - *"How do I return this sweater that doesn't fit? Can I drop it off at Whole Foods?"*
  - *"Tracking shows the return package was received at your warehouse 5 days ago, where is my refund?"*
  - *"I don't have a printer to print the return shipping label."*
- **Negative / Boundary Examples**:
  - *"I was charged twice for my Prime membership renewal."* &rarr; `subscription_and_billing`.
  - *"The product arrived broken so I want a refund."* &rarr; `damaged_or_wrong_item`.
- **Expected Action**:
  - Explain standard return drop-off procedures (label-free, box-free return QR codes).
  - Clarify standard refund processing timelines (3-5 business days after inspection).
  - Escalate if customer disputes a missing refund beyond the SLA.

---

### 4. `account_security_and_login`
- **Definition**: Critical situations involving unauthorized account access, password reset failures, two-factor authentication (OTP) bypass issues, suspicious login alerts, or potential phishing emails.
- **Positive Examples**:
  - *"Someone placed 5 orders on my account using a stolen credit card in another country!"*
  - *"I am locked out because OTP codes are being sent to an old phone number I no longer own."*
  - *"Received an email claiming my account is suspended asking me to click a link. Is this legit?"*
- **Negative / Boundary Examples**:
  - *"How do I change my default shipping address?"* &rarr; `other` / account preferences.
  - *"My credit card was declined at checkout."* &rarr; `subscription_and_billing` or `other`.
- **Expected Action**:
  - **MANDATORY ESCALATION**: Automated systems must NEVER attempt to reset credentials or verify identity in open channels.
  - Direct the customer exclusively to official two-step verification and account recovery portals.
  - Urge immediate suspension of compromised payment instruments if active fraud is reported.

---

### 5. `subscription_and_billing`
- **Definition**: Questions or disputes related to Amazon Prime membership fees, recurring digital subscriptions (Kindle Unlimited, Music Unlimited, Prime Video channels), double billings, or unexpected credit card statements.
- **Positive Examples**:
  - *"I noticed a $14.99 charge on my credit card statement for Amazon Digital that I did not authorize."*
  - *"How do I cancel my Amazon Prime annual subscription before it auto-renews?"*
  - *"I was billed twice for my Audible membership this billing cycle."*
- **Negative / Boundary Examples**:
  - *"Where is the refund for the pair of shoes I returned last week?"* &rarr; `return_and_refund`.
  - *"My gift card balance is not applying to my cart."* &rarr; `other`.
- **Expected Action**:
  - Provide navigational path to *Manage Your Prime Membership* / *Memberships & Subscriptions*.
  - Detail standard cancellation and prorated refund rules.
  - Escalate to human billing specialists if unauthorized or fraudulent transactions are suspected.

---

### 6. `product_technical_issue`
- **Definition**: Glitches, setup difficulties, firmware update errors, or operational failures involving Amazon devices (Kindle e-readers, Echo smart speakers, Fire TV sticks, Blink/Ring security cameras) or digital media apps.
- **Positive Examples**:
  - *"My Fire TV stick is stuck on the loading logo and rebooting continuously."*
  - *"Echo Dot says 'I'm having trouble understanding right now' every time I speak."*
  - *"Kindle Paperwhite screen is frozen on the screensaver and not responding to the power button."*
- **Negative / Boundary Examples**:
  - *"The screen on my Kindle arrived cracked out of the box."* &rarr; `damaged_or_wrong_item`.
  - *"The Prime Video app says my subscription expired."* &rarr; `subscription_and_billing`.
- **Expected Action**:
  - Provide standard step-by-step triage (power cycle, 40-second hard reset, Wi-Fi restart).
  - Link to official Amazon Device Support documentation.

---

### 7. `feedback_or_complaint`
- **Definition**: Customer venting or formal grievances regarding courier behavior (delivery driver threw package, blocked driveway), inappropriate packaging (excessive plastic, unpadded envelope), rude phone support agent, or general company policies.
- **Positive Examples**:
  - *"The delivery driver threw my package over the 6-foot fence onto the concrete driveway."*
  - *"Why did you ship a tiny memory card in an enormous cardboard box filled with plastic bubbles?"*
  - *"Your phone agent hung up on me after keeping me on hold for 45 minutes!"*
- **Negative / Boundary Examples**:
  - *"The driver threw my package and the glass inside shattered."* &rarr; `damaged_or_wrong_item` (primary physical resolution needed).
  - *"I am threatening legal action unless my refund is processed immediately."* &rarr; Escalate under legal / high-risk policy.
- **Expected Action**:
  - Acknowledge frustration with genuine empathy and professional de-escalation tone.
  - Assure customer that delivery feedback is logged directly against the carrier route.
  - Escalate if complaint involves driver misconduct or harassment.

---

### 8. `other`
- **Definition**: Greetings, conversational chitchat, inquiries outside Amazon's service scope, non-English tweets, ambiguous messages lacking sufficient context, or general website navigational help.
- **Positive Examples**:
  - *"Hello, are you guys open today?"*
  - *"Can you tell me if the new iPhone is better than Samsung Galaxy?"*
  - *"Thanks for the great help earlier, have a wonderful weekend!"*
- **Negative / Boundary Examples**:
  - Any query that clearly fits one of the 7 primary domain categories above.
- **Expected Action**:
  - Provide polite, helpful clarification request or friendly closure.
  - For ambiguous queries, ask clarifying questions: *"Could you please share more details about your order or account inquiry so we can guide you accurately?"*
