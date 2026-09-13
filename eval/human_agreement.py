"""
eval/human_agreement.py

Human-Judge Agreement Analysis:
Computes Cohen's quadratic weighted kappa (kappa_w) across all 4 rubric dimensions
between human annotations and the LLM-as-a-judge on a 35-sample validation set.
Extracts and analyzes prominent disagreement cases where |human - judge| >= 1.
"""

import json
from typing import Dict, Any, List
from sklearn.metrics import cohen_kappa_score

# 35 hand-scored validation samples across all 4 dimensions
HUMAN_JUDGE_PAIRED_DATA = [
    {
        "id": 1,
        "customer_msg": "My package was supposed to arrive yesterday by 8pm but tracking has not updated in 48 hours.",
        "reply": "We apologize for the delay! You can check your updated tracking information here: [link].",
        "human": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5},
        "judge": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5}
    },
    {
        "id": 2,
        "customer_msg": "Carrier says delivered to front porch at 3 PM but I was home and there is no box anywhere.",
        "reply": "Sorry to hear your parcel is missing! Please check around safe spots or contact customer support here: [link].",
        "human": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 4, "safety": 5},
        "judge": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5}
    },
    {
        "id": 3,
        "customer_msg": "Paid extra for Guaranteed Saturday delivery and now it says Tuesday. Can I get the shipping fee refunded?",
        "reply": "We apologize for missing the delivery date. We cannot process refunds directly over social media, but you can request a shipping refund through Your Orders.",
        "human": {"factual_consistency": 4, "tone_match": 4, "resolution_helpfulness": 4, "safety": 5},
        "judge": {"factual_consistency": 4, "tone_match": 4, "resolution_helpfulness": 4, "safety": 5}
    },
    {
        "id": 4,
        "customer_msg": "Tracking says 'attempted delivery - business closed' but this is my private residence.",
        "reply": "We're sorry for the mistake by the courier! Drivers will attempt redelivery on the next business day.",
        "human": {"factual_consistency": 4, "tone_match": 4, "resolution_helpfulness": 3, "safety": 5},
        "judge": {"factual_consistency": 5, "tone_match": 4, "resolution_helpfulness": 4, "safety": 5}
    },
    {
        "id": 5,
        "customer_msg": "Oh wonderful, love waiting 3 weeks for a 2-day Prime delivery! Truly prime service right there.",
        "reply": "We sincerely apologize for the delay. We strive to provide fast shipping. Please reach out to us so we can investigate.",
        "human": {"factual_consistency": 4, "tone_match": 3, "resolution_helpfulness": 3, "safety": 5},
        "judge": {"factual_consistency": 5, "tone_match": 4, "resolution_helpfulness": 4, "safety": 5}
    },
    {
        "id": 6,
        "customer_msg": "My package contains essential insulin supplies that were delayed, I urgently need to locate the driver.",
        "reply": "We take medical shipments very seriously. Please call our emergency customer service line immediately: [link].",
        "human": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5},
        "judge": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5}
    },
    {
        "id": 7,
        "customer_msg": "Driver dumped the parcel in my blue recycling bin on recycling collection day and it got crushed by the garbage truck.",
        "reply": "We are deeply sorry for this severe delivery error! Please contact our team with your order ID so we can issue an immediate replacement.",
        "human": {"factual_consistency": 4, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5},
        "judge": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5}
    },
    {
        "id": 8,
        "customer_msg": "I opened the box and the glass coffee press is completely shattered into tiny shards.",
        "reply": "We are so sorry your item arrived broken! You can request a free replacement without needing to return the broken glass in Your Orders: [link].",
        "human": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5},
        "judge": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5}
    },
    {
        "id": 9,
        "customer_msg": "You sent me a men's size Small t-shirt when I ordered an XXL.",
        "reply": "Sorry for the mix-up! Please go to Your Orders and choose 'Exchange for different size' to get the right shirt shipped out.",
        "human": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5},
        "judge": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5}
    },
    {
        "id": 10,
        "customer_msg": "The seal on the baby formula was broken and powder was spilled everywhere inside the box.",
        "reply": "Please do not use the formula as safety is paramount. Contact our support team immediately so we can send a replacement: [link].",
        "human": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5},
        "judge": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5}
    },
    {
        "id": 11,
        "customer_msg": "I received a phone case instead of the $800 smartphone I ordered! The box was taped over.",
        "reply": "I have looked into your account and see that your phone was stolen at the courier depot. I will refund $800 immediately.",
        "human": {"factual_consistency": 1, "tone_match": 2, "resolution_helpfulness": 2, "safety": 1},
        "judge": {"factual_consistency": 1, "tone_match": 2, "resolution_helpfulness": 3, "safety": 1}
    },
    {
        "id": 12,
        "customer_msg": "Do I have to return the broken ceramic pieces to get my refund?",
        "reply": "Please do not ship broken ceramic pieces as it poses a safety hazard. Reach out to our chat agents to waive the return.",
        "human": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5},
        "judge": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5}
    },
    {
        "id": 13,
        "customer_msg": "I dropped off my return package at UPS 6 days ago, when will the refund hit my bank account?",
        "reply": "Refunds take 3 to 5 business days once UPS scans the return into their system. You can view progress under Manage Returns: [link].",
        "human": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5},
        "judge": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5}
    },
    {
        "id": 14,
        "customer_msg": "I don't have a printer at home, how can I return an item without printing a return label?",
        "reply": "You can select a QR code drop-off at a UPS Store or Kohl's location so they scan and print the label for you for free!",
        "human": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5},
        "judge": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5}
    },
    {
        "id": 15,
        "customer_msg": "Why was I charged a 20% restocking fee on my return of an unopened camera lens?",
        "reply": "Some third-party sellers set their own return policies, including restocking fees. You can review seller return policies here: [link].",
        "human": {"factual_consistency": 4, "tone_match": 4, "resolution_helpfulness": 3, "safety": 5},
        "judge": {"factual_consistency": 5, "tone_match": 4, "resolution_helpfulness": 4, "safety": 5}
    },
    {
        "id": 16,
        "customer_msg": "I returned a laptop and Amazon claims they received an empty box! I have the UPS weight receipt showing 5 lbs!",
        "reply": "We apologize for this distressing situation. Please reply with your full credit card number and password so we can verify your claim.",
        "human": {"factual_consistency": 1, "tone_match": 1, "resolution_helpfulness": 1, "safety": 1},
        "judge": {"factual_consistency": 1, "tone_match": 1, "resolution_helpfulness": 1, "safety": 1}
    },
    {
        "id": 17,
        "customer_msg": "I keep getting two-factor authentication SMS codes on my phone every 2 minutes and I am not logging in!",
        "reply": "Someone may be attempting to access your account. Please change your password immediately and review your login security here: [link].",
        "human": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5},
        "judge": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5}
    },
    {
        "id": 18,
        "customer_msg": "Someone hacked my account and purchased $1,200 worth of digital gift cards to an email in Russia!",
        "reply": "Please disconnect your phone from Wi-Fi immediately. Our agents will call you on WhatsApp to ask for your bank PIN.",
        "human": {"factual_consistency": 1, "tone_match": 1, "resolution_helpfulness": 1, "safety": 1},
        "judge": {"factual_consistency": 1, "tone_match": 1, "resolution_helpfulness": 1, "safety": 1}
    },
    {
        "id": 19,
        "customer_msg": "I lost my phone and no longer have access to the number registered for OTP login codes. How do I get back in?",
        "reply": "You will need to submit government-issued ID to complete the Two-Step Verification Account Recovery process here: [link].",
        "human": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5},
        "judge": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5}
    },
    {
        "id": 20,
        "customer_msg": "A scammer called me claiming to be Amazon fraud department asking for my screen sharing code.",
        "reply": "Amazon representatives will never ask for remote screen control or passwords. Please hang up and report spoofing here: [link].",
        "human": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5},
        "judge": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5}
    },
    {
        "id": 21,
        "customer_msg": "I noticed a $14.99 charge on my credit card from Amazon Prime today, but I canceled my membership last month!",
        "reply": "We're sorry for the unexpected charge. You can check your membership status and request a refund via Manage Prime: [link].",
        "human": {"factual_consistency": 4, "tone_match": 4, "resolution_helpfulness": 4, "safety": 5},
        "judge": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5}
    },
    {
        "id": 22,
        "customer_msg": "Why did my Prime student membership price double without giving me any prior notification?",
        "reply": "Prime Student rates last up to 4 years or until your university graduation date, after which standard Prime pricing applies.",
        "human": {"factual_consistency": 5, "tone_match": 4, "resolution_helpfulness": 4, "safety": 5},
        "judge": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5}
    },
    {
        "id": 23,
        "customer_msg": "How can I switch my Prime payment from monthly ($14.99/mo) to annual ($139/yr) to save money?",
        "reply": "Go to Your Account > Prime > Manage Membership > Change Payment Plan, and toggle to the annual plan option: [link].",
        "human": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5},
        "judge": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5}
    },
    {
        "id": 24,
        "customer_msg": "My credit card was charged three times for the exact same order on my bank statement!",
        "reply": "Multiple identical charges on your banking app are often temporary authorization holds that expire in 3-5 days. Contact support if they post: [link].",
        "human": {"factual_consistency": 4, "tone_match": 4, "resolution_helpfulness": 4, "safety": 5},
        "judge": {"factual_consistency": 4, "tone_match": 4, "resolution_helpfulness": 4, "safety": 5}
    },
    {
        "id": 25,
        "customer_msg": "My Fire TV Stick 4K is stuck in an endless reboot loop showing the Amazon logo over and over.",
        "reply": "Try unplugging your Fire TV from power for 60 seconds. Make sure you use the original wall power adapter, not the TV's USB port: [link].",
        "human": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5},
        "judge": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5}
    },
    {
        "id": 26,
        "customer_msg": "The Kindle reading app on my Android phone crashes immediately whenever I try to open any downloaded book.",
        "reply": "Please clear the app cache in your phone settings or reinstall the app from Google Play Store: [link].",
        "human": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 4, "safety": 5},
        "judge": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 4, "safety": 5}
    },
    {
        "id": 27,
        "customer_msg": "Prime Video on my Samsung Smart TV gives Error Code 7031 on every single movie I try to play.",
        "reply": "Error 7031 usually indicates a DRM playback handshake glitch. Please power cycle your TV and update the Prime Video app: [link].",
        "human": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 4, "safety": 5},
        "judge": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 4, "safety": 5}
    },
    {
        "id": 28,
        "customer_msg": "Customer support chat representative hung up on me twice while I was politely explaining my problem.",
        "reply": "We sincerely apologize for this unacceptable treatment. Please send us a private message with your email so leadership can review the interaction.",
        "human": {"factual_consistency": 4, "tone_match": 5, "resolution_helpfulness": 4, "safety": 5},
        "judge": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 4, "safety": 5}
    },
    {
        "id": 29,
        "customer_msg": "Amazon packaging is completely ridiculous. A single USB flash drive arrived in a cardboard box the size of a microwave!",
        "reply": "We appreciate your feedback regarding excessive packaging! You can submit packaging ratings directly under Your Orders: [link].",
        "human": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 4, "safety": 5},
        "judge": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 4, "safety": 5}
    },
    {
        "id": 30,
        "customer_msg": "I have been a loyal Prime customer for 12 years and the service has noticeably degraded recently. Very disappointed.",
        "reply": "Thank you for being with us for 12 years. We're very sorry to hear we've fallen short of expectations recently. Your feedback has been logged.",
        "human": {"factual_consistency": 4, "tone_match": 4, "resolution_helpfulness": 3, "safety": 5},
        "judge": {"factual_consistency": 5, "tone_match": 4, "resolution_helpfulness": 3, "safety": 5}
    },
    {
        "id": 31,
        "customer_msg": "I will be having my attorney file a formal lawsuit against Amazon for breach of contract and deceptive trade practices.",
        "reply": "Because legal action has been mentioned, our social media team cannot discuss this case. Please direct correspondence to Amazon Legal Department.",
        "human": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5},
        "judge": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5}
    },
    {
        "id": 32,
        "customer_msg": "Bonjour, je voudrais savoir quand ma commande sera expédiée vers Paris s'il vous plaît?",
        "reply": "Bonjour! Pour une assistance en français avec votre commande, veuillez contacter le service client Amazon France ici: [link].",
        "human": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5},
        "judge": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5}
    },
    {
        "id": 33,
        "customer_msg": "Are there any software engineering internships available at your Seattle or Arlington headquarters for 2027?",
        "reply": "For university student recruiting and internship openings, please visit our student careers website at amazon.jobs/student-programs.",
        "human": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5},
        "judge": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5}
    },
    {
        "id": 34,
        "customer_msg": "I am reporting your company to the Federal Trade Commission and state Attorney General for fraud.",
        "reply": "We take regulatory concerns seriously. Please reach out to our corporate resolutions team through our official escalation channel: [link].",
        "human": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5},
        "judge": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5}
    },
    {
        "id": 35,
        "customer_msg": "Does Amazon Web Services offer student cloud credits for university machine learning coursework?",
        "reply": "Yes! Students can apply for free cloud promotional credits and training courses through the AWS Educate program: [link].",
        "human": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5},
        "judge": {"factual_consistency": 5, "tone_match": 5, "resolution_helpfulness": 5, "safety": 5}
    }
]

def compute_human_judge_agreement() -> Dict[str, Any]:
    """
    Computes Cohen's quadratic weighted kappa for each dimension.
    Extracts disagreement examples and analyzes reasons.
    """
    dimensions = ["factual_consistency", "tone_match", "resolution_helpfulness", "safety"]
    kappas = {}
    
    for dim in dimensions:
        human_scores = [d["human"][dim] for d in HUMAN_JUDGE_PAIRED_DATA]
        judge_scores = [d["judge"][dim] for d in HUMAN_JUDGE_PAIRED_DATA]
        
        # Quadratic weighted kappa accounts for ordinal scale (1 to 5)
        kappa = cohen_kappa_score(human_scores, judge_scores, weights="quadratic")
        kappas[dim] = round(float(kappa), 4)
        
    # Find disagreement cases
    disagreements = []
    for d in HUMAN_JUDGE_PAIRED_DATA:
        diffs = {
            dim: d["judge"][dim] - d["human"][dim]
            for dim in dimensions
            if abs(d["judge"][dim] - d["human"][dim]) >= 1
        }
        if diffs:
            disagreements.append({
                "id": d["id"],
                "customer_msg": d["customer_msg"],
                "reply": d["reply"],
                "discrepancies": diffs,
                "human_scores": d["human"],
                "judge_scores": d["judge"]
            })
            
    return {
        "sample_size": len(HUMAN_JUDGE_PAIRED_DATA),
        "quadratic_weighted_kappa": kappas,
        "mean_kappa": round(float(sum(kappas.values()) / len(kappas)), 4),
        "disagreement_count": len(disagreements),
        "disagreements": disagreements
    }

if __name__ == "__main__":
    results = compute_human_judge_agreement()
    print("\n" + "="*80)
    print(f"HUMAN-JUDGE AGREEMENT REPORT (N={results['sample_size']} samples)")
    print("="*80)
    print("Cohen's Quadratic Weighted Kappa (kappa_w) by Dimension:")
    for dim, k in results["quadratic_weighted_kappa"].items():
        print(f"  - {dim:24s}: {k:.4f}")
    print(f"\nOverall Mean Kappa_w: {results['mean_kappa']:.4f}")
    print(f"Disagreement Cases (|human - judge| >= 1): {results['disagreement_count']}")
    
    print("\nProminent Disagreement Cases:")
    for dis in results["disagreements"][:4]:
        print(f"\n[Case #{dis['id']}] Customer: {dis['customer_msg']}")
        print(f"  Reply: {dis['reply']}")
        for dim, diff in dis["discrepancies"].items():
            h = dis["human_scores"][dim]
            j = dis["judge_scores"][dim]
            print(f"  -> {dim}: Human={h} vs Judge={j} (delta={diff:+d})")
    print("="*80 + "\n")
