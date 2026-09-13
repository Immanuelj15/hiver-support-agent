"""
scripts/generate_golden_set.py

Builds the 150-sample stratified golden evaluation set (golden_set/golden_150.jsonl)
covering all 8 intents with realistic edge cases (sarcasm, multi-issue, venting,
non-actionable, and high-risk security/legal threats).
"""

import json
from pathlib import Path

GOLDEN_ITEMS = [
    # -------------------------------------------------------------
    # 1. ORDER_DELIVERY_DELAY (24 items)
    # -------------------------------------------------------------
    {
        "customer_msg": "My package was supposed to arrive yesterday by 8pm but tracking has not updated in 48 hours.",
        "gold_intent": "order_delivery_delay",
        "gold_should_escalate": False,
        "gold_reason_notes": "Standard delivery tracking inquiry, safe for automated tracking link and self-serve guidance.",
        "reference_reply": "We apologize for the shipping delay. You can view the most up-to-date tracking details under Your Orders here: [link]."
    },
    {
        "customer_msg": "Carrier says delivered to front porch at 3 PM but I was home and there is no box anywhere.",
        "gold_intent": "order_delivery_delay",
        "gold_should_escalate": False,
        "gold_reason_notes": "Standard false delivery scan; advise checking safe spots and waiting 24h or contacting support via secure link.",
        "reference_reply": "Sorry to hear your package hasn't turned up! Sometimes carriers mark parcels early. Please check safe locations or contact us here: [link]."
    },
    {
        "customer_msg": "Paid extra for Guaranteed Saturday delivery and now it says Tuesday. Can I get the shipping fee refunded?",
        "gold_intent": "order_delivery_delay",
        "gold_should_escalate": True,
        "gold_reason_notes": "Explicit request for financial shipping refund requiring account verification.",
        "reference_reply": "We're very sorry for missing the guaranteed delivery date. Please message us your order ID privately so an agent can process your shipping refund."
    },
    {
        "customer_msg": "Tracking says 'attempted delivery - business closed' but this is my private residence and I was in the living room.",
        "gold_intent": "order_delivery_delay",
        "gold_should_escalate": False,
        "gold_reason_notes": "Carrier access issue, provide redelivery guidance.",
        "reference_reply": "We apologize for the delivery issue. Couriers will typically re-attempt on the next business day. You can update delivery instructions here: [link]."
    },
    {
        "customer_msg": "Oh wonderful, love waiting 3 weeks for a 2-day Prime delivery! Truly prime service right there.",
        "gold_intent": "order_delivery_delay",
        "gold_should_escalate": False,
        "gold_reason_notes": "Sarcastic complaint regarding delay; needs empathetic response and tracking link.",
        "reference_reply": "We sincerely apologize for the delay and frustration! Please check your order status here: [link] or send us a DM so we can look into it."
    },
    {
        "customer_msg": "Where is tracking number TBA123456789012? It has been sitting at the facility since Monday.",
        "gold_intent": "order_delivery_delay",
        "gold_should_escalate": False,
        "gold_reason_notes": "Direct tracking status query.",
        "reference_reply": "We're sorry for the pause in transit updates. You can track live updates directly in Your Orders: [link]."
    },
    {
        "customer_msg": "My package contains essential insulin supplies that were delayed, I urgently need to locate the driver.",
        "gold_intent": "order_delivery_delay",
        "gold_should_escalate": True,
        "gold_reason_notes": "Urgent medical/safety implication requiring immediate human dispatcher intervention.",
        "reference_reply": "We take medical shipments very seriously. Please contact our live escalation team immediately by phone or chat here: [link]."
    },
    {
        "customer_msg": "Is there any delay affecting deliveries in the North London area due to weather conditions?",
        "gold_intent": "order_delivery_delay",
        "gold_should_escalate": False,
        "gold_reason_notes": "General informational question regarding weather delays.",
        "reference_reply": "Adverse weather can occasionally impact local logistics. Check your individual order tracking for real-time estimated arrival: [link]."
    },
    {
        "customer_msg": "My delivery date keeps getting pushed back every single morning. First Wednesday, then Friday, now Monday.",
        "gold_intent": "order_delivery_delay",
        "gold_should_escalate": False,
        "gold_reason_notes": "Repeated rolling delay inquiry.",
        "reference_reply": "We understand how frustrating date changes are. Please reach out to our team directly through our help center to investigate: [link]."
    },
    {
        "customer_msg": "Driver left package in the communal hallway and it got stolen before I got home from work.",
        "gold_intent": "order_delivery_delay",
        "gold_should_escalate": True,
        "gold_reason_notes": "Stolen package / porch piracy requiring replacement or claim investigation.",
        "reference_reply": "We are so sorry your parcel went missing from the hallway. Please DM us your order details or reach out to customer service here: [link]."
    },
    {
        "customer_msg": "Can I redirect my shipment to an Amazon Locker instead since the delivery has been delayed?",
        "gold_intent": "order_delivery_delay",
        "gold_should_escalate": False,
        "gold_reason_notes": "Policy inquiry on Locker redirection during transit.",
        "reference_reply": "Once an item is in transit, the destination address usually cannot be modified. You can view your options in Your Orders: [link]."
    },
    {
        "customer_msg": "My order says 'Package transferred to local post office for final delivery'. What does that mean?",
        "gold_intent": "order_delivery_delay",
        "gold_should_escalate": False,
        "gold_reason_notes": "Informational inquiry regarding postal carrier handoff.",
        "reference_reply": "This means your local postal carrier will complete the final mile delivery to your mailbox or door, usually within 1-2 business days."
    },
    {
        "customer_msg": "Ordered birthday gifts on Prime 2 days ago and they have not even dispatched yet. Birthday is tomorrow!",
        "gold_intent": "order_delivery_delay",
        "gold_should_escalate": False,
        "gold_reason_notes": "Pending dispatch timeline inquiry.",
        "reference_reply": "We know how important birthdays are! Items often dispatch from nearby facilities close to the delivery date. Check live status here: [link]."
    },
    {
        "customer_msg": "Package delayed due to incorrect address on label. How do I fix the house number?",
        "gold_intent": "order_delivery_delay",
        "gold_should_escalate": True,
        "gold_reason_notes": "Address correction requiring account-level intervention or carrier re-routing.",
        "reference_reply": "To update address details for an active delivery, please contact our support team immediately through our secure portal: [link]."
    },
    {
        "customer_msg": "Tracking shows 'held at customs' for my international import order. Who pays the duty fees?",
        "gold_intent": "order_delivery_delay",
        "gold_should_escalate": False,
        "gold_reason_notes": "Import fees deposit policy clarification.",
        "reference_reply": "Import Fees Deposit details are calculated at checkout. You can review customs clearance guidelines here: [link]."
    },
    {
        "customer_msg": "My parcel was delivered to a neighbor two streets away according to the courier photo. I don't know who lives there.",
        "gold_intent": "order_delivery_delay",
        "gold_should_escalate": True,
        "gold_reason_notes": "Misdelivery to unknown third party requiring investigation or reshipment.",
        "reference_reply": "We apologize for the misdelivery! Please reach out to our customer care team with your order ID so we can issue a replacement or refund: [link]."
    },
    {
        "customer_msg": "Why does Amazon logistics show 'Out for delivery' at 7am but it still hasn't arrived by 9pm?",
        "gold_intent": "order_delivery_delay",
        "gold_should_escalate": False,
        "gold_reason_notes": "Late evening delivery window explanation.",
        "reference_reply": "Drivers frequently deliver until 9 PM or 10 PM. If your order does not arrive tonight, please check tracking in the morning: [link]."
    },
    {
        "customer_msg": "Three separate items from the same order shipped in three different boxes on different days. Why?",
        "gold_intent": "order_delivery_delay",
        "gold_should_escalate": False,
        "gold_reason_notes": "Multi-package shipment logistics explanation.",
        "reference_reply": "Items often ship from different fulfillment centers to get them to you as quickly as possible. You can track each parcel individually in Your Orders."
    },
    {
        "customer_msg": "Tracking states parcel damaged in transit and returning to sender. What happens now?",
        "gold_intent": "order_delivery_delay",
        "gold_should_escalate": True,
        "gold_reason_notes": "Return to sender due to carrier damage requiring refund or replacement trigger.",
        "reference_reply": "When a package is returned damaged by carrier, a full refund is automatically issued once received, or you can contact us to reorder: [link]."
    },
    {
        "customer_msg": "It has been 10 days since the expected delivery date. The system won't let me request a refund online.",
        "gold_intent": "order_delivery_delay",
        "gold_should_escalate": True,
        "gold_reason_notes": "System lock preventing customer refund after long delay; manual agent override required.",
        "reference_reply": "We apologize for the delay. Since the online option is unavailable, please connect with a live agent here to assist: [link]."
    },
    {
        "customer_msg": "Driver dumped the parcel in my blue recycling bin on recycling collection day and it got crushed by the garbage truck.",
        "gold_intent": "order_delivery_delay",
        "gold_should_escalate": True,
        "gold_reason_notes": "Severe driver error resulting in total item loss, requiring claim and escalation.",
        "reference_reply": "We are terribly sorry about this delivery error! Please message us privately or reach out here so we can resolve this right away: [link]."
    },
    {
        "customer_msg": "Can I pay extra right now to upgrade an already dispatched shipment to same-day delivery?",
        "gold_intent": "order_delivery_delay",
        "gold_should_escalate": False,
        "gold_reason_notes": "Shipping speed modification policy question.",
        "reference_reply": "Unfortunately, once an order has dispatched from our fulfillment center, the shipping speed cannot be upgraded."
    },
    {
        "customer_msg": "The tracking status has been showing 'Departed facility' for 5 days with zero checkpoint scans.",
        "gold_intent": "order_delivery_delay",
        "gold_should_escalate": False,
        "gold_reason_notes": "Long gap between carrier scans.",
        "reference_reply": "Packages in long-distance transit may not scan until reaching the destination hub. If not delivered by the target date, contact us: [link]."
    },
    {
        "customer_msg": "I ordered fresh groceries and the delivery window was 2 hours ago. The ice cream is going to melt!",
        "gold_intent": "order_delivery_delay",
        "gold_should_escalate": True,
        "gold_reason_notes": "Perishable grocery delivery delay with imminent product destruction.",
        "reference_reply": "We are very sorry for the grocery delay. Please contact Amazon Fresh customer support immediately for an immediate refund or redelivery: [link]."
    },

    # -------------------------------------------------------------
    # 2. DAMAGED_OR_WRONG_ITEM (20 items)
    # -------------------------------------------------------------
    {
        "customer_msg": "I opened the box and the glass coffee press is completely shattered into tiny shards.",
        "gold_intent": "damaged_or_wrong_item",
        "gold_should_escalate": False,
        "gold_reason_notes": "Standard damaged item claim covered by automated return/replacement flow.",
        "reference_reply": "We're so sorry your coffee press arrived broken! You can easily set up a free replacement or refund in Your Orders here: [link]."
    },
    {
        "customer_msg": "You sent me a men's size Small t-shirt when I ordered an XXL. The barcode label on the plastic bag was wrong.",
        "gold_intent": "damaged_or_wrong_item",
        "gold_should_escalate": False,
        "gold_reason_notes": "Mislabeled inventory item, standard return/replacement procedure.",
        "reference_reply": "We apologize for the mix-up with sizing! Please initiate an exchange for the correct size via Your Orders: [link]."
    },
    {
        "customer_msg": "The seal on the baby formula was broken and powder was spilled everywhere inside the box.",
        "gold_intent": "damaged_or_wrong_item",
        "gold_should_escalate": True,
        "gold_reason_notes": "Safety/health risk regarding infant nutrition product.",
        "reference_reply": "We apologize and take infant product safety extremely seriously. Please do not consume the product and contact our team immediately: [link]."
    },
    {
        "customer_msg": "Ordered 4 dining chairs and only received 3 in the shipment. The packing slip says 4.",
        "gold_intent": "damaged_or_wrong_item",
        "gold_should_escalate": True,
        "gold_reason_notes": "Missing part of a multi-piece set requiring warehouse inventory lookup.",
        "reference_reply": "We're sorry for the missing chair! Please message us your order details or contact our support team here to send the remaining item: [link]."
    },
    {
        "customer_msg": "Bought a brand new SSD hard drive and it arrived in an unsealed box with dust on it. It looks used!",
        "gold_intent": "damaged_or_wrong_item",
        "gold_should_escalate": False,
        "gold_reason_notes": "Item condition complaint (used sold as new), standard return flow.",
        "reference_reply": "We expect all new items to arrive in pristine condition. Please request a replacement through Your Orders: [link]."
    },
    {
        "customer_msg": "The shampoo bottle leaked all over the books that were packed in the same box, ruining them.",
        "gold_intent": "damaged_or_wrong_item",
        "gold_should_escalate": True,
        "gold_reason_notes": "Multiple damaged items from leakage requiring composite claim handling.",
        "reference_reply": "We are so sorry about the poor packaging! Please reach out to customer service here so we can replace both the shampoo and the books: [link]."
    },
    {
        "customer_msg": "I received a phone case instead of the $800 smartphone I ordered! The box was taped over.",
        "gold_intent": "damaged_or_wrong_item",
        "gold_should_escalate": True,
        "gold_reason_notes": "High-value misdelivery/theft suspicion requiring fraud investigation.",
        "reference_reply": "We take this matter very seriously. Please contact our specialized investigation team immediately via phone or chat: [link]."
    },
    {
        "customer_msg": "The TV screen is completely cracked internally. When I turned it on, half the display is black lines.",
        "gold_intent": "damaged_or_wrong_item",
        "gold_should_escalate": False,
        "gold_reason_notes": "Standard large item damaged on arrival, eligible for carrier pickup return.",
        "reference_reply": "We apologize for the damaged television. You can schedule a free carrier pickup and replacement via Your Orders: [link]."
    },
    {
        "customer_msg": "The shoes came with two left feet! Both are size 9 left shoes.",
        "gold_intent": "damaged_or_wrong_item",
        "gold_should_escalate": False,
        "gold_reason_notes": "Humorous manufacturing defect, standard exchange flow.",
        "reference_reply": "We're sorry for the mix-up! Please head to Your Orders to request an exchange for a proper pair: [link]."
    },
    {
        "customer_msg": "Do I have to return the broken ceramic pieces to get my refund? It is dangerous to pack glass shards.",
        "gold_intent": "damaged_or_wrong_item",
        "gold_should_escalate": True,
        "gold_reason_notes": "Hazardous return exception request; safety policy waiver required.",
        "reference_reply": "You should not mail broken glass. Please contact our support team directly so an agent can waive the return requirement: [link]."
    },
    {
        "customer_msg": "I ordered an English version of the board game and you sent the German edition.",
        "gold_intent": "damaged_or_wrong_item",
        "gold_should_escalate": False,
        "gold_reason_notes": "Wrong language variant received, standard replacement.",
        "reference_reply": "We apologize for the language mix-up. Please return the item for a replacement in English via Your Orders: [link]."
    },
    {
        "customer_msg": "The box arrived looking like an accordion and the computer monitor inside was snapped in half.",
        "gold_intent": "damaged_or_wrong_item",
        "gold_should_escalate": False,
        "gold_reason_notes": "Transit damage, standard return/replacement link.",
        "reference_reply": "We are very sorry your monitor arrived damaged! You can arrange an immediate return and replacement in Your Orders: [link]."
    },
    {
        "customer_msg": "Received an item with an expired 'Best By' date from six months ago.",
        "gold_intent": "damaged_or_wrong_item",
        "gold_should_escalate": False,
        "gold_reason_notes": "Expired perishable product, automated refund eligible.",
        "reference_reply": "We apologize for sending an expired item. Please visit Your Orders to receive an automatic refund or replacement: [link]."
    },
    {
        "customer_msg": "The jacket is missing the detachable hood that was described on the product page.",
        "gold_intent": "damaged_or_wrong_item",
        "gold_should_escalate": False,
        "gold_reason_notes": "Missing accessory from product bundle.",
        "reference_reply": "Sorry to hear the hood was missing! You can request a replacement or return the item via Your Orders: [link]."
    },
    {
        "customer_msg": "Ordered a pair of gold earrings and received a packet of screws instead.",
        "gold_intent": "damaged_or_wrong_item",
        "gold_should_escalate": True,
        "gold_reason_notes": "Complete merchandise mismatch on jewelry item requiring inventory tag check.",
        "reference_reply": "We apologize for this unexpected mix-up! Please reach out to our team here with your order number so we can fix this: [link]."
    },
    {
        "customer_msg": "The power supply cable for the vacuum cleaner is frayed with exposed copper wire.",
        "gold_intent": "damaged_or_wrong_item",
        "gold_should_escalate": True,
        "gold_reason_notes": "Electrical safety hazard requiring escalation.",
        "reference_reply": "Please unplug and do not use the item due to safety risks. Contact our support team immediately for assistance: [link]."
    },
    {
        "customer_msg": "The book spine was completely split and several pages were torn out.",
        "gold_intent": "damaged_or_wrong_item",
        "gold_should_escalate": False,
        "gold_reason_notes": "Damaged book condition, standard replacement flow.",
        "reference_reply": "We apologize for the damaged book. You can easily request a replacement copy at no cost via Your Orders: [link]."
    },
    {
        "customer_msg": "You sent me a refurbished laptop when the listing clearly said brand new in box.",
        "gold_intent": "damaged_or_wrong_item",
        "gold_should_escalate": True,
        "gold_reason_notes": "Seller misrepresentation / fulfillment error requiring price difference or return.",
        "reference_reply": "We apologize for the discrepancy. Please contact our support team so we can investigate the seller and arrange a return: [link]."
    },
    {
        "customer_msg": "The perfume bottle cap was loose and the entire contents evaporated and soaked into the packing paper.",
        "gold_intent": "damaged_or_wrong_item",
        "gold_should_escalate": False,
        "gold_reason_notes": "Spilled cosmetics product, automated return/refund flow.",
        "reference_reply": "We're sorry your perfume spilled in transit! You can request a refund or replacement under Your Orders here: [link]."
    },
    {
        "customer_msg": "Only received the left joycon controller, the right one was not in the package.",
        "gold_intent": "damaged_or_wrong_item",
        "gold_should_escalate": False,
        "gold_reason_notes": "Incomplete pair of items, standard replacement flow.",
        "reference_reply": "We're sorry one controller was missing. You can process an exchange for a complete set through Your Orders: [link]."
    },

    # -------------------------------------------------------------
    # 3. RETURN_AND_REFUND (22 items)
    # -------------------------------------------------------------
    {
        "customer_msg": "I dropped off my return package at UPS 6 days ago, when will the refund hit my bank account?",
        "gold_intent": "return_and_refund",
        "gold_should_escalate": False,
        "gold_reason_notes": "Standard return processing timeline inquiry.",
        "reference_reply": "Refunds typically process within 3-5 business days after the carrier scans the return. You can check return status here: [link]."
    },
    {
        "customer_msg": "I don't have a printer at home, how can I return an item without printing a return label?",
        "gold_intent": "return_and_refund",
        "gold_should_escalate": False,
        "gold_reason_notes": "Self-serve printerless return (QR code) policy guidance.",
        "reference_reply": "You can select a printerless return option (such as UPS Store or Kohl's QR code) during the return process in Your Orders."
    },
    {
        "customer_msg": "I canceled my order 15 minutes after placing it. Why is the money still showing as deducted from my card?",
        "gold_intent": "return_and_refund",
        "gold_should_escalate": False,
        "gold_reason_notes": "Authorization hold vs charge explanation.",
        "reference_reply": "When an order is canceled promptly, the pending bank authorization hold typically drops off within a few business days depending on your bank."
    },
    {
        "customer_msg": "Why was I charged a 20% restocking fee on my return of an unopened camera lens?",
        "gold_intent": "return_and_refund",
        "gold_should_escalate": True,
        "gold_reason_notes": "Third-party seller fee dispute requiring human agent override.",
        "reference_reply": "Third-party sellers may have specific return policies. Please reach out to our team with your order ID so we can review the fee: [link]."
    },
    {
        "customer_msg": "My return window closed two days ago, can you grant a brief exception so I can return this unopened gift?",
        "gold_intent": "return_and_refund",
        "gold_should_escalate": True,
        "gold_reason_notes": "Discretionary policy exception requiring human supervisor authorization.",
        "reference_reply": "Return window extensions are evaluated on a case-by-case basis. Please connect with our support agents here: [link]."
    },
    {
        "customer_msg": "The delivery driver never came to pick up the return package today as scheduled. How do I rebook?",
        "gold_intent": "return_and_refund",
        "gold_should_escalate": False,
        "gold_reason_notes": "Rescheduling carrier pickup flow.",
        "reference_reply": "We apologize for the missed pickup! You can easily reschedule a pickup date in the Manage Returns section of Your Orders: [link]."
    },
    {
        "customer_msg": "I selected refund to Amazon gift card balance but I meant to choose my original credit card. Can you switch it?",
        "gold_intent": "return_and_refund",
        "gold_should_escalate": True,
        "gold_reason_notes": "Refund method conversion requiring accounting modification.",
        "reference_reply": "Once a gift card refund is issued, changing payment methods requires agent assistance. Please reach out to us here: [link]."
    },
    {
        "customer_msg": "Can I return an opened mattress under the 100-night trial policy?",
        "gold_intent": "return_and_refund",
        "gold_should_escalate": False,
        "gold_reason_notes": "Informational inquiry on mattress return policy.",
        "reference_reply": "Mattress returns under sleep trials usually involve scheduling a special bulky item pickup. Details are available under Your Orders: [link]."
    },
    {
        "customer_msg": "UPS tracking shows my return parcel was delivered to your warehouse two weeks ago, but my account still says 'waiting for return'.",
        "gold_intent": "return_and_refund",
        "gold_should_escalate": True,
        "gold_reason_notes": "Warehouse intake bottleneck requiring manual refund release.",
        "reference_reply": "We apologize for the delay in processing your return at our warehouse. Please contact us with your tracking receipt to issue your refund: [link]."
    },
    {
        "customer_msg": "Can I combine two returns from different orders into one single box to save packaging?",
        "gold_intent": "return_and_refund",
        "gold_should_escalate": False,
        "gold_reason_notes": "Policy guidance against combining separate return labels.",
        "reference_reply": "We recommend sending returns in separate boxes with their specific return labels to ensure each item is tracked and refunded correctly."
    },
    {
        "customer_msg": "I received an email stating my refund was processed, but it has been 10 business days and my bank sees nothing.",
        "gold_intent": "return_and_refund",
        "gold_should_escalate": True,
        "gold_reason_notes": "Payment gateway / bank transfer failure requiring ARN trace number.",
        "reference_reply": "If your refund has not appeared after 10 business days, please contact our support team so we can provide an Acquirer Reference Number (ARN): [link]."
    },
    {
        "customer_msg": "How do I return a hazardous material item like a lithium battery or bottle of lighter fluid?",
        "gold_intent": "return_and_refund",
        "gold_should_escalate": False,
        "gold_reason_notes": "Hazmat return policy guidance.",
        "reference_reply": "Certain hazmat items cannot be returned via standard mail. When initiating a return in Your Orders, special disposal or refund instructions will appear."
    },
    {
        "customer_msg": "I returned a laptop and Amazon claims they received an empty box! I have the UPS weight receipt showing 5 lbs!",
        "gold_intent": "return_and_refund",
        "gold_should_escalate": True,
        "gold_reason_notes": "Fraud allegation / high-value return dispute requiring warehouse weight audit.",
        "reference_reply": "We take weight discrepancies seriously. Please contact our leadership escalations team with your UPS drop-off weight receipt here: [link]."
    },
    {
        "customer_msg": "Are return shipping fees deducted from my refund if the item was sold by a third party?",
        "gold_intent": "return_and_refund",
        "gold_should_escalate": False,
        "gold_reason_notes": "Third-party seller return policy explanation.",
        "reference_reply": "Third-party seller returns may incur return shipping costs unless the return is due to seller error or defect. Check policy details here: [link]."
    },
    {
        "customer_msg": "I moved to a new address. How do I get my replacement order sent to my new house?",
        "gold_intent": "return_and_refund",
        "gold_should_escalate": True,
        "gold_reason_notes": "Address override on replacement order requiring customer service intervention.",
        "reference_reply": "To redirect a replacement to a new address, please contact our support specialists before the replacement ships: [link]."
    },
    {
        "customer_msg": "How long do I have to drop off my return at Whole Foods after generating the QR code?",
        "gold_intent": "return_and_refund",
        "gold_should_escalate": False,
        "gold_reason_notes": "Informational return QR code validity window (typically 30 days).",
        "reference_reply": "Return QR codes are typically valid for 30 days from creation. You can view or regenerate your code in Your Orders: [link]."
    },
    {
        "customer_msg": "I bought this with promotional credit. If I return the item, do I get the promo credit back?",
        "gold_intent": "return_and_refund",
        "gold_should_escalate": False,
        "gold_reason_notes": "Promotional coupon refund terms.",
        "reference_reply": "Promotional credits are usually one-time use and non-refundable, but any out-of-pocket amount paid will be returned to your original payment method."
    },
    {
        "customer_msg": "My account was refunded only $12 when the total receipt was $45. Why the difference?",
        "gold_intent": "return_and_refund",
        "gold_should_escalate": True,
        "gold_reason_notes": "Partial refund discrepancy requiring account audit.",
        "reference_reply": "We apologize for the refund discrepancy. Please reach out to our team with your order details so we can investigate the difference: [link]."
    },
    {
        "customer_msg": "Is there an option to have the refund credited instantly before I even drop off the package?",
        "gold_intent": "return_and_refund",
        "gold_should_escalate": False,
        "gold_reason_notes": "Explanation of Advance Refund eligibility.",
        "reference_reply": "Instant refunds are automatically offered on select items based on account standing, but usually require a carrier first-scan."
    },
    {
        "customer_msg": "Can I return digital software or video game download codes if I haven't redeemed them yet?",
        "gold_intent": "return_and_refund",
        "gold_should_escalate": False,
        "gold_reason_notes": "Digital item non-returnable policy explanation.",
        "reference_reply": "Digital downloads, games, and software keys are generally non-returnable and non-refundable once delivered. See digital terms here: [link]."
    },
    {
        "customer_msg": "I returned 3 items in the same box using the combined QR code, but only 1 has been refunded.",
        "gold_intent": "return_and_refund",
        "gold_should_escalate": True,
        "gold_reason_notes": "Multi-item consolidation discrepancy requiring manual item reconciliation.",
        "reference_reply": "Items bundled together can sometimes take longer to be processed individually. Please message our team so we can confirm the other items: [link]."
    },
    {
        "customer_msg": "How can I check the remaining balance on an Amazon return gift card?",
        "gold_intent": "return_and_refund",
        "gold_should_escalate": False,
        "gold_reason_notes": "Gift card balance check self-service guidance.",
        "reference_reply": "You can view your current Gift Card balance anytime by visiting Your Account > Gift Cards: [link]."
    },

    # -------------------------------------------------------------
    # 4. ACCOUNT_SECURITY_AND_LOGIN (18 items) - ALL ESCALATE
    # -------------------------------------------------------------
    {
        "customer_msg": "I keep getting two-factor authentication SMS codes on my phone every 2 minutes and I am not logging in!",
        "gold_intent": "account_security_and_login",
        "gold_should_escalate": True,
        "gold_reason_notes": "Active brute-force or unauthorized credential stuffing attempt.",
        "reference_reply": "Please secure your account immediately by resetting your password and reviewing your 2SV settings here: [link]."
    },
    {
        "customer_msg": "Someone hacked my account and purchased $1,200 worth of digital gift cards to an email in Russia!",
        "gold_intent": "account_security_and_login",
        "gold_should_escalate": True,
        "gold_reason_notes": "Fraudulent unauthorized transactions, immediate account freeze needed.",
        "reference_reply": "We take unauthorized access very seriously. Please contact our specialized Account Security team immediately here: [link]."
    },
    {
        "customer_msg": "I cannot sign into my Amazon account. It says my email is not registered, but I've had it for 10 years!",
        "gold_intent": "account_security_and_login",
        "gold_should_escalate": True,
        "gold_reason_notes": "Primary email hijacked or account takeover.",
        "reference_reply": "It sounds like your account credentials may have been altered. Please reach out to our security team for account verification: [link]."
    },
    {
        "customer_msg": "I lost my phone and no longer have access to the number registered for OTP login codes. How do I get back in?",
        "gold_intent": "account_security_and_login",
        "gold_should_escalate": True,
        "gold_reason_notes": "Two-Step Verification recovery requiring government ID identity proofing.",
        "reference_reply": "You can begin the Two-Step Verification account recovery process by uploading verification documents here: [link]."
    },
    {
        "customer_msg": "I received an email saying my password was successfully reset, but I did not initiate this.",
        "gold_intent": "account_security_and_login",
        "gold_should_escalate": True,
        "gold_reason_notes": "Unauthorized credential reset.",
        "reference_reply": "If you did not request a password reset, please secure your account immediately and reach out to our team: [link]."
    },
    {
        "customer_msg": "How do I revoke access from old devices and log out of all active Amazon sessions?",
        "gold_intent": "account_security_and_login",
        "gold_should_escalate": True,
        "gold_reason_notes": "Session revocation and device security.",
        "reference_reply": "You can manage registered devices and deregister sessions under Manage Your Content and Devices: [link]."
    },
    {
        "customer_msg": "Suspicious email claiming to be from Amazon Billing stating my order will be canceled unless I update my banking info.",
        "gold_intent": "account_security_and_login",
        "gold_should_escalate": True,
        "gold_reason_notes": "Phishing reporting and account validation.",
        "reference_reply": "Do not click any links in suspicious emails. You can report phishing directly to stop-spoofing@amazon.com."
    },
    {
        "customer_msg": "My Amazon account has been placed on hold and I was asked to submit a utility bill to unfreeze it.",
        "gold_intent": "account_security_and_login",
        "gold_should_escalate": True,
        "gold_reason_notes": "Account compliance hold requiring specialist review.",
        "reference_reply": "Account holds require direct verification with our Account Specialist team. Please follow the instructions sent via email."
    },
    {
        "customer_msg": "Can someone help me set up an Authenticator App instead of SMS for my 2-factor authentication?",
        "gold_intent": "account_security_and_login",
        "gold_should_escalate": True,
        "gold_reason_notes": "Security configuration assistance.",
        "reference_reply": "You can configure an Authenticator App by navigating to Login & Security > Two-Step Verification Settings: [link]."
    },
    {
        "customer_msg": "There is an unknown delivery address in Florida saved on my account that I never added.",
        "gold_intent": "account_security_and_login",
        "gold_should_escalate": True,
        "gold_reason_notes": "Compromised address book, potential brushing or account breach.",
        "reference_reply": "Please delete any unfamiliar addresses under Your Addresses and change your password immediately: [link]."
    },
    {
        "customer_msg": "A scammer called me claiming to be Amazon fraud department asking for my screen sharing code.",
        "gold_intent": "account_security_and_login",
        "gold_should_escalate": True,
        "gold_reason_notes": "Impersonation scam / social engineering attack.",
        "reference_reply": "Amazon will never call asking for screen access or passwords. Please hang up and report the incident here: [link]."
    },
    {
        "customer_msg": "I was locked out because I entered my password wrong 5 times. How long does the lockout last?",
        "gold_intent": "account_security_and_login",
        "gold_should_escalate": True,
        "gold_reason_notes": "Account lockout duration and unlock flow.",
        "reference_reply": "Temporary lockouts usually expire after a few hours, or you can use the 'Forgot Password' link to reset access."
    },
    {
        "customer_msg": "Someone created a fake seller account using my personal national identity number and credit card.",
        "gold_intent": "account_security_and_login",
        "gold_should_escalate": True,
        "gold_reason_notes": "Identity theft and unauthorized merchant registration.",
        "reference_reply": "Please contact our identity verification and seller fraud team immediately to initiate an investigation: [link]."
    },
    {
        "customer_msg": "How do I close and permanently delete my Amazon account and all associated personal data?",
        "gold_intent": "account_security_and_login",
        "gold_should_escalate": True,
        "gold_reason_notes": "Permanent account deletion / privacy data erasure.",
        "reference_reply": "To permanently close your account and delete your data, please submit a request via Close Your Amazon Account: [link]."
    },
    {
        "customer_msg": "My teen was using my account and accidentally purchased $300 of in-game currency on Fire Tablet.",
        "gold_intent": "account_security_and_login",
        "gold_should_escalate": True,
        "gold_reason_notes": "Unauthorized in-app purchases by minor, parental controls review.",
        "reference_reply": "Please reach out to our digital support team to review unauthorized in-app purchases and enable parental PIN controls: [link]."
    },
    {
        "customer_msg": "Can you check if my credit card information has been leaked in any recent database breaches?",
        "gold_intent": "account_security_and_login",
        "gold_should_escalate": True,
        "gold_reason_notes": "Security breach inquiry, must be handled with care.",
        "reference_reply": "Amazon maintains strict encryption and security standards. Please review account security best practices here: [link]."
    },
    {
        "customer_msg": "I am unable to change my email address because the verification code is sent to an email that is now deleted.",
        "gold_intent": "account_security_and_login",
        "gold_should_escalate": True,
        "gold_reason_notes": "Orphaned email address preventing self-service verification.",
        "reference_reply": "Please contact our customer support team directly so an agent can verify your identity and update your account: [link]."
    },
    {
        "customer_msg": "My account was compromised and the hacker set up a passkey that prevents me from recovering it.",
        "gold_intent": "account_security_and_login",
        "gold_should_escalate": True,
        "gold_reason_notes": "Adversarial credential takeover using FIDO passkeys.",
        "reference_reply": "Please connect with our account recovery specialists immediately to begin manual identity verification: [link]."
    },

    # -------------------------------------------------------------
    # 5. SUBSCRIPTION_AND_BILLING (20 items)
    # -------------------------------------------------------------
    {
        "customer_msg": "I noticed a $14.99 charge on my credit card from Amazon Prime today, but I canceled my membership last month!",
        "gold_intent": "subscription_and_billing",
        "gold_should_escalate": True,
        "gold_reason_notes": "Unexpected subscription charge requiring billing verification and refund.",
        "reference_reply": "We apologize for the unexpected charge. Please check Manage Prime Membership or reach out via DM so we can assist with a refund: [link]."
    },
    {
        "customer_msg": "Why did my Prime student membership price double without giving me any prior notification?",
        "gold_intent": "subscription_and_billing",
        "gold_should_escalate": False,
        "gold_reason_notes": "Prime Student graduation transition policy explanation.",
        "reference_reply": "Prime Student discounted rates run for up to 4 years or until graduation, after which it transitions to standard Prime."
    },
    {
        "customer_msg": "I was billed for an Amazon Music Unlimited family plan that I never authorized or signed up for.",
        "gold_intent": "subscription_and_billing",
        "gold_should_escalate": True,
        "gold_reason_notes": "Unrecognized recurring subscription charge requiring cancellation and credit.",
        "reference_reply": "We're sorry for the surprise subscription. You can cancel active channels in Your Memberships & Subscriptions or contact us: [link]."
    },
    {
        "customer_msg": "How can I switch my Prime payment from monthly ($14.99/mo) to annual ($139/yr) to save money?",
        "gold_intent": "subscription_and_billing",
        "gold_should_escalate": False,
        "gold_reason_notes": "Standard self-serve membership billing plan change.",
        "reference_reply": "You can easily switch billing plans by going to Your Account > Prime > Manage Membership > Change Payment Plan: [link]."
    },
    {
        "customer_msg": "My credit card was charged three times for the exact same order on my bank statement!",
        "gold_intent": "subscription_and_billing",
        "gold_should_escalate": True,
        "gold_reason_notes": "Multiple billing transactions / duplicate charge dispute.",
        "reference_reply": "We apologize for the multiple charges. Please connect with our billing specialists so we can verify if they are pending authorizations: [link]."
    },
    {
        "customer_msg": "How do I cancel my Kindle Unlimited free trial before it automatically renews next week?",
        "gold_intent": "subscription_and_billing",
        "gold_should_escalate": False,
        "gold_reason_notes": "Self-service trial cancellation guidance.",
        "reference_reply": "You can cancel anytime before renewal by visiting Your Memberships & Subscriptions and selecting Cancel Kindle Unlimited: [link]."
    },
    {
        "customer_msg": "Where can I download VAT or sales tax invoices for my business purchases from last quarter?",
        "gold_intent": "subscription_and_billing",
        "gold_should_escalate": False,
        "gold_reason_notes": "Self-service tax invoice download instructions.",
        "reference_reply": "You can print or download official tax invoices for any past order under Your Orders > Invoice / Order Details: [link]."
    },
    {
        "customer_msg": "My credit card expired and my Prime renewal failed. How do I update my payment method without losing my benefits?",
        "gold_intent": "subscription_and_billing",
        "gold_should_escalate": False,
        "gold_reason_notes": "Payment method update instructions.",
        "reference_reply": "You can update your card details in Your Account > Manage Prime Membership > Update Payment Method: [link]."
    },
    {
        "customer_msg": "Why was my account charged $9.99 from 'AMZN DIGITAL' when I haven't purchased anything digital recently?",
        "gold_intent": "subscription_and_billing",
        "gold_should_escalate": True,
        "gold_reason_notes": "Ambiguous digital charge lookup requiring transaction audit.",
        "reference_reply": "You can view what digital item or channel subscription caused the charge by visiting Your Digital Orders: [link]."
    },
    {
        "customer_msg": "Can I share my Amazon Prime shipping benefits with my spouse through Amazon Household?",
        "gold_intent": "subscription_and_billing",
        "gold_should_escalate": False,
        "gold_reason_notes": "Informational Amazon Household policy question.",
        "reference_reply": "Yes! You can link two adult accounts to share Prime shipping, Prime Video, and digital content via Amazon Household: [link]."
    },
    {
        "customer_msg": "I am being charged for Paramount+ through Prime Video but I already pay Paramount directly on their app.",
        "gold_intent": "subscription_and_billing",
        "gold_should_escalate": False,
        "gold_reason_notes": "Prime Video Channels cancellation guidance.",
        "reference_reply": "Prime Video Channels are separate from standalone app subscriptions. You can cancel the Prime channel under Manage Your Subscriptions: [link]."
    },
    {
        "customer_msg": "How do I remove an old expired debit card from my Amazon wallet?",
        "gold_intent": "subscription_and_billing",
        "gold_should_escalate": False,
        "gold_reason_notes": "Self-service payment wallet management.",
        "reference_reply": "You can remove saved cards anytime by navigating to Your Account > Your Payments and selecting Remove: [link]."
    },
    {
        "customer_msg": "I was billed for Prime even though I never used the benefits this year. Can I get a full refund?",
        "gold_intent": "subscription_and_billing",
        "gold_should_escalate": True,
        "gold_reason_notes": "Full annual refund evaluation based on benefit non-usage.",
        "reference_reply": "Members who have not used their Prime benefits are usually eligible for a full refund upon cancellation. Please contact support: [link]."
    },
    {
        "customer_msg": "Does Amazon accept payment via PayPal or Apple Pay for regular marketplace orders?",
        "gold_intent": "subscription_and_billing",
        "gold_should_escalate": False,
        "gold_reason_notes": "Supported payment methods policy question.",
        "reference_reply": "Amazon does not directly accept PayPal or Apple Pay, but accepts major credit/debit cards, Amazon Pay, and gift cards."
    },
    {
        "customer_msg": "Why was my order canceled due to payment decline when my bank says they never received a charge request?",
        "gold_intent": "subscription_and_billing",
        "gold_should_escalate": True,
        "gold_reason_notes": "Payment gateway processing failure requiring backend inspection.",
        "reference_reply": "We apologize for the payment issue. Please contact our support specialists so we can check the transaction status: [link]."
    },
    {
        "customer_msg": "Can I pay for my Prime membership using an Amazon gift card balance?",
        "gold_intent": "subscription_and_billing",
        "gold_should_escalate": False,
        "gold_reason_notes": "Gift card usage for Prime membership eligibility.",
        "reference_reply": "In most regions, gift card balances cannot be used for recurring auto-renewing Prime memberships unless a backup card is on file."
    },
    {
        "customer_msg": "I received an unexpected charge of $1.00 on my card from Amazon. What is this for?",
        "gold_intent": "subscription_and_billing",
        "gold_should_escalate": False,
        "gold_reason_notes": "Standard temporary card authorization explanation.",
        "reference_reply": "A $1.00 transaction is typically a temporary authorization hold to verify your card is active, and will disappear shortly."
    },
    {
        "customer_msg": "How do I pause my Audible subscription without losing my accumulated audio credits?",
        "gold_intent": "subscription_and_billing",
        "gold_should_escalate": False,
        "gold_reason_notes": "Audible membership pause policy guidance.",
        "reference_reply": "You can put your Audible membership on hold for up to 3 months once a year while keeping your unused credits: [link]."
    },
    {
        "customer_msg": "My Amazon Store Card has fraudulent interest charges applied, who do I contact?",
        "gold_intent": "subscription_and_billing",
        "gold_should_escalate": True,
        "gold_reason_notes": "Synchrony Bank / co-branded credit card dispute.",
        "reference_reply": "For Amazon Store Card financing and interest inquiries, please contact Synchrony Bank customer care directly."
    },
    {
        "customer_msg": "Can I split a single order payment between two different credit cards?",
        "gold_intent": "subscription_and_billing",
        "gold_should_escalate": False,
        "gold_reason_notes": "Split payment limitation policy question.",
        "reference_reply": "Amazon does not support splitting payments across two credit cards, but you can combine one credit card with an Amazon Gift Card."
    },

    # -------------------------------------------------------------
    # 6. PRODUCT_TECHNICAL_ISSUE (18 items)
    # -------------------------------------------------------------
    {
        "customer_msg": "My Fire TV Stick 4K is stuck in an endless reboot loop showing the Amazon logo over and over.",
        "gold_intent": "product_technical_issue",
        "gold_should_escalate": False,
        "gold_reason_notes": "Standard Fire TV troubleshooting steps (power cycle, original adapter).",
        "reference_reply": "Try unplugging your Fire TV from power for 60 seconds and ensure it is plugged into a wall outlet rather than a TV USB port: [link]."
    },
    {
        "customer_msg": "The Kindle reading app on my Android phone crashes immediately whenever I try to open any downloaded book.",
        "gold_intent": "product_technical_issue",
        "gold_should_escalate": False,
        "gold_reason_notes": "App crash troubleshooting (cache clear, reinstall).",
        "reference_reply": "Please try clearing the Kindle app cache in your phone settings or reinstalling the application to resolve crash issues: [link]."
    },
    {
        "customer_msg": "My Echo Dot (4th Gen) has a solid red ring and Alexa says 'I'm having trouble connecting to the internet'.",
        "gold_intent": "product_technical_issue",
        "gold_should_escalate": False,
        "gold_reason_notes": "Echo Wi-Fi troubleshooting and microphone mute check.",
        "reference_reply": "A red ring means the microphone is muted or Wi-Fi is disconnected. Unplug the device for 30 seconds and check Wi-Fi in the Alexa app."
    },
    {
        "customer_msg": "Prime Video on my Samsung Smart TV gives Error Code 7031 on every single movie I try to play.",
        "gold_intent": "product_technical_issue",
        "gold_should_escalate": False,
        "gold_reason_notes": "Prime Video error 7031 DRM/streaming troubleshooting.",
        "reference_reply": "Error 7031 is usually resolved by restarting your smart TV, updating the Prime Video app, or power-cycling your router: [link]."
    },
    {
        "customer_msg": "How do I factory reset an Echo Show 8 before giving it to a friend?",
        "gold_intent": "product_technical_issue",
        "gold_should_escalate": False,
        "gold_reason_notes": "Self-service factory reset instructions.",
        "reference_reply": "To factory reset, swipe down from the top of the screen, tap Settings > Device Options > Reset to Factory Defaults."
    },
    {
        "customer_msg": "My Kindle Paperwhite screen is frozen on the screensaver and holding the power button does nothing.",
        "gold_intent": "product_technical_issue",
        "gold_should_escalate": False,
        "gold_reason_notes": "Hard reboot procedure for E-ink Kindle.",
        "reference_reply": "Hold the power button down continuously for a full 40 seconds until the device reboots, then plug it into a wall charger for 30 minutes."
    },
    {
        "customer_msg": "Alexa smart home routines are not triggering my Philips Hue smart bulbs anymore.",
        "gold_intent": "product_technical_issue",
        "gold_should_escalate": False,
        "gold_reason_notes": "Smart home skill re-linking troubleshooting.",
        "reference_reply": "Try disabling and re-enabling the Hue skill in the Alexa app, then run device discovery again: [link]."
    },
    {
        "customer_msg": "Fire Tablet HD 10 battery drains from 100% to zero in less than one hour even when in standby.",
        "gold_intent": "product_technical_issue",
        "gold_should_escalate": True,
        "gold_reason_notes": "Defective lithium battery requiring warranty hardware replacement.",
        "reference_reply": "Abnormal battery drain may indicate a hardware fault. Please contact our device support team for warranty replacement: [link]."
    },
    {
        "customer_msg": "Why does Prime Video say 'This title is not available in your location' when I am streaming from home?",
        "gold_intent": "product_technical_issue",
        "gold_should_escalate": False,
        "gold_reason_notes": "Geo-IP / VPN detection explanation.",
        "reference_reply": "Make sure any VPN or proxy services are disabled, and verify your country settings under Manage Your Content and Devices: [link]."
    },
    {
        "customer_msg": "How can I transfer sideloaded PDF files onto my Kindle Oasis?",
        "gold_intent": "product_technical_issue",
        "gold_should_escalate": False,
        "gold_reason_notes": "Send to Kindle service instructions.",
        "reference_reply": "You can send PDFs directly to your Kindle using the Send to Kindle web tool or by emailing your @kindle.com address: [link]."
    },
    {
        "customer_msg": "My Fire TV remote completely stopped responding even after putting in brand new fresh batteries.",
        "gold_intent": "product_technical_issue",
        "gold_should_escalate": False,
        "gold_reason_notes": "Remote re-pairing sequence guidance.",
        "reference_reply": "Try resetting your remote: hold the Left button, Menu button, and Back button simultaneously for 12 seconds to re-pair."
    },
    {
        "customer_msg": "Echo Studio makes a loud popping and buzzing static sound whenever playing music at medium volume.",
        "gold_intent": "product_technical_issue",
        "gold_should_escalate": True,
        "gold_reason_notes": "Hardware speaker blow-out requiring warranty claim.",
        "reference_reply": "Hardware distortion may require a replacement. Please contact our Echo technical support team here: [link]."
    },
    {
        "customer_msg": "How do I turn off motion tracking on my Echo Show 10 screen?",
        "gold_intent": "product_technical_issue",
        "gold_should_escalate": False,
        "gold_reason_notes": "Device motion settings configuration.",
        "reference_reply": "You can turn off motion by swiping down from the top and tapping the Motion button, or by saying 'Alexa, turn off motion'."
    },
    {
        "customer_msg": "Audio is completely out of sync with video when streaming 4K HDR content on the Prime Video Windows app.",
        "gold_intent": "product_technical_issue",
        "gold_should_escalate": False,
        "gold_reason_notes": "App hardware acceleration settings check.",
        "reference_reply": "Check for audio driver updates on your PC and try disabling hardware acceleration in the Prime Video app settings."
    },
    {
        "customer_msg": "My Kindle Oasis charging port is loose and only charges if the cable is held at a specific angle.",
        "gold_intent": "product_technical_issue",
        "gold_should_escalate": True,
        "gold_reason_notes": "Physical USB port damage requiring repair or warranty replacement.",
        "reference_reply": "A loose charging port indicates physical hardware fatigue. Please reach out to our device specialists for repair options: [link]."
    },
    {
        "customer_msg": "Can I connect two Echo Dots together to create stereo left and right channel audio?",
        "gold_intent": "product_technical_issue",
        "gold_should_escalate": False,
        "gold_reason_notes": "Stereo speaker pair setup in Alexa app.",
        "reference_reply": "Yes! In the Alexa app, go to Devices > '+' > Combine Speakers > Stereo Pair, and select your two compatible Echo devices."
    },
    {
        "customer_msg": "Why does my Fire TV stick display 'Low Storage Space' when I only have 3 apps installed?",
        "gold_intent": "product_technical_issue",
        "gold_should_escalate": False,
        "gold_reason_notes": "System data bloat cleanup instructions.",
        "reference_reply": "Navigate to Settings > Applications > Manage Installed Applications to clear data on streaming apps that cache media files."
    },
    {
        "customer_msg": "The touchscreen on my Fire 7 tablet has a dead zone in the center that does not register touches.",
        "gold_intent": "product_technical_issue",
        "gold_should_escalate": True,
        "gold_reason_notes": "Digitizer hardware failure requiring replacement.",
        "reference_reply": "A dead touchscreen area is a digitizer hardware issue. Please reach out to customer support to initiate a warranty replacement: [link]."
    },

    # -------------------------------------------------------------
    # 7. FEEDBACK_OR_COMPLAINT (16 items)
    # -------------------------------------------------------------
    {
        "customer_msg": "Your delivery driver threw my heavy package over my 7-foot locked gate, smashing my flower pots!",
        "gold_intent": "feedback_or_complaint",
        "gold_should_escalate": True,
        "gold_reason_notes": "Property damage by logistics partner requiring courier incident report.",
        "reference_reply": "We take property damage very seriously and sincerely apologize. Please DM us your delivery details so we can investigate the driver."
    },
    {
        "customer_msg": "Customer support chat representative hung up on me twice while I was politely explaining my problem.",
        "gold_intent": "feedback_or_complaint",
        "gold_should_escalate": True,
        "gold_reason_notes": "Severe agent misconduct / service quality complaint.",
        "reference_reply": "We expect our representatives to treat all customers with utmost respect. Please DM us your chat timestamp so we can coach the team."
    },
    {
        "customer_msg": "Amazon packaging is completely ridiculous. A single USB flash drive arrived in a cardboard box the size of a microwave!",
        "gold_intent": "feedback_or_complaint",
        "gold_should_escalate": False,
        "gold_reason_notes": "General packaging feedback, non-actionable venting.",
        "reference_reply": "Thank you for sharing your feedback on packaging waste! You can submit direct packaging ratings in Your Orders to help us improve."
    },
    {
        "customer_msg": "I have been a loyal Prime customer for 12 years and the service has noticeably degraded recently. Very disappointed.",
        "gold_intent": "feedback_or_complaint",
        "gold_should_escalate": False,
        "gold_reason_notes": "General customer sentiment venting without specific order inquiry.",
        "reference_reply": "We truly appreciate your 12 years of loyalty and are sorry to hear we've let you down. We're constantly working to improve our service."
    },
    {
        "customer_msg": "Delivery driver walked right across my freshly seeded lawn instead of using the paved walkway.",
        "gold_intent": "feedback_or_complaint",
        "gold_should_escalate": False,
        "gold_reason_notes": "Driver behavioral feedback, advise updating delivery instructions.",
        "reference_reply": "We apologize for the driver's actions! You can add permanent delivery instructions to your address profile in Your Account: [link]."
    },
    {
        "customer_msg": "Your automated phone bot is an infuriating circular loop that refuses to transfer me to a human!",
        "gold_intent": "feedback_or_complaint",
        "gold_should_escalate": False,
        "gold_reason_notes": "IVR frustration, provide direct click-to-call link.",
        "reference_reply": "We understand the frustration! You can request an immediate callback from a human agent without waiting on hold here: [link]."
    },
    {
        "customer_msg": "Driver left my expensive laptop right in the pouring rain on the front steps without even knocking.",
        "gold_intent": "feedback_or_complaint",
        "gold_should_escalate": True,
        "gold_reason_notes": "Negligent delivery handling endangering high-value electronics.",
        "reference_reply": "We are very sorry your package was exposed to the weather! Please check the contents and message us if anything is damaged: [link]."
    },
    {
        "customer_msg": "Why do Amazon delivery vans constantly block entire bike lanes and park on pedestrian crosswalks in my neighborhood?",
        "gold_intent": "feedback_or_complaint",
        "gold_should_escalate": False,
        "gold_reason_notes": "Community fleet complaint, provide logistics feedback link.",
        "reference_reply": "Safety in the community is a top priority for our logistics partners. Please report vehicle details and license plates here: [link]."
    },
    {
        "customer_msg": "The search results on Amazon are 90% sponsored ads and cheap copycat brands now. It is impossible to find quality items.",
        "gold_intent": "feedback_or_complaint",
        "gold_should_escalate": False,
        "gold_reason_notes": "Marketplace UI and ad placement feedback.",
        "reference_reply": "Thank you for your feedback regarding search results. We're actively working on search quality and appreciate your perspective."
    },
    {
        "customer_msg": "Your driver honked their horn repeatedly outside my house at 6:00 AM on a Sunday morning waking the entire baby nursery.",
        "gold_intent": "feedback_or_complaint",
        "gold_should_escalate": True,
        "gold_reason_notes": "Noise violation / courier disturbance complaint.",
        "reference_reply": "We sincerely apologize for the early morning disturbance! Please share your tracking number privately so we can address this driver."
    },
    {
        "customer_msg": "Received an item covered in sticky black grease on the outside of the shipping box. Got on my carpet.",
        "gold_intent": "feedback_or_complaint",
        "gold_should_escalate": True,
        "gold_reason_notes": "Carpet staining / property damage from dirty transit parcel.",
        "reference_reply": "We apologize for the condition of the box and your carpet! Please contact customer care so we can assist with a claim: [link]."
    },
    {
        "customer_msg": "Third time this month customer care gave me contradictory information. Train your representatives better!",
        "gold_intent": "feedback_or_complaint",
        "gold_should_escalate": False,
        "gold_reason_notes": "Training and support quality feedback.",
        "reference_reply": "We apologize for the conflicting information you received. If you still need help with an ongoing issue, please let us know."
    },
    {
        "customer_msg": "Delivery instructions specifically said 'leave at side door' and the driver dumped it by the front street curb.",
        "gold_intent": "feedback_or_complaint",
        "gold_should_escalate": False,
        "gold_reason_notes": "Ignored delivery instructions, standard guidance.",
        "reference_reply": "We're sorry your delivery instructions were overlooked! You can rate the delivery and leave driver feedback in Your Orders."
    },
    {
        "customer_msg": "Customer support took 45 minutes to answer my call on hold. What happened to your quick service?",
        "gold_intent": "feedback_or_complaint",
        "gold_should_escalate": False,
        "gold_reason_notes": "Long queue hold time complaint.",
        "reference_reply": "We apologize for the long wait time during peak hours. For faster service, our callback feature can call you directly: [link]."
    },
    {
        "customer_msg": "Your return process requires me to drive 20 miles to the nearest drop-off location. That is unacceptable.",
        "gold_intent": "feedback_or_complaint",
        "gold_should_escalate": False,
        "gold_reason_notes": "Rural return location complaint, suggest pickup options.",
        "reference_reply": "We understand the inconvenience! If drop-off points are far, you can check for carrier home pickup options in Your Orders."
    },
    {
        "customer_msg": "Amazon has gone downhill since Jeff Bezos stepped down as CEO. Zero quality control anymore.",
        "gold_intent": "feedback_or_complaint",
        "gold_should_escalate": False,
        "gold_reason_notes": "General executive/company sentiment venting.",
        "reference_reply": "We appreciate you sharing your thoughts and are committed to earning your trust with every order."
    },

    # -------------------------------------------------------------
    # 8. OTHER (12 items) - ALL ESCALATE
    # -------------------------------------------------------------
    {
        "customer_msg": "I will be having my attorney file a formal lawsuit against Amazon for breach of contract and deceptive trade practices.",
        "gold_intent": "other",
        "gold_should_escalate": True,
        "gold_reason_notes": "Explicit legal threat requiring immediate referral to corporate legal counsel.",
        "reference_reply": "As you have indicated legal action, we must direct all further communications to Amazon Legal Department."
    },
    {
        "customer_msg": "Bonjour, je voudrais savoir quand ma commande sera expédiée vers Paris s'il vous plaît?",
        "gold_intent": "other",
        "gold_should_escalate": True,
        "gold_reason_notes": "Non-English message (French), must route to Amazon.fr French language support queue.",
        "reference_reply": "Bonjour, pour toute assistance en français, veuillez contacter le service client Amazon.fr ici: [link]."
    },
    {
        "customer_msg": "Hola buenas tardes, no he recibido mi pedido en Madrid y el repartidor no contesta el teléfono.",
        "gold_intent": "other",
        "gold_should_escalate": True,
        "gold_reason_notes": "Non-English message (Spanish), route to Amazon.es support team.",
        "reference_reply": "Hola, para asistencia en español por favor comuníquese con atención al cliente de Amazon.es aquí: [link]."
    },
    {
        "customer_msg": "Are there any software engineering internships available at your Seattle or Arlington headquarters for 2027?",
        "gold_intent": "other",
        "gold_should_escalate": True,
        "gold_reason_notes": "Recruiting / employment inquiry outside customer service scope.",
        "reference_reply": "For job opportunities and student internships, please visit our official careers portal at amazon.jobs."
    },
    {
        "customer_msg": "Who is the senior VP in charge of customer experience and what is their direct executive email address?",
        "gold_intent": "other",
        "gold_should_escalate": True,
        "gold_reason_notes": "Executive escalation request.",
        "reference_reply": "For executive escalations, please submit your inquiry through our customer service escalation channel: [link]."
    },
    {
        "customer_msg": "I am a journalist with the New York Times writing a story on holiday retail logistics, who can I speak to for press comment?",
        "gold_intent": "other",
        "gold_should_escalate": True,
        "gold_reason_notes": "Media / PR press inquiry requiring corporate communications transfer.",
        "reference_reply": "For press inquiries, please contact Amazon Public Relations directly at amazon-pr@amazon.com."
    },
    {
        "customer_msg": "Amazon is an evil monopoly destroying small mom and pop shops worldwide. You should all be broken up by antitrust regulators!",
        "gold_intent": "other",
        "gold_should_escalate": True,
        "gold_reason_notes": "Political / antitrust rant with no customer service inquiry.",
        "reference_reply": "Thank you for sharing your perspective on retail and marketplace policies."
    },
    {
        "customer_msg": "Can I buy wholesale bulk quantities of pallets directly from Amazon liquidation warehouses?",
        "gold_intent": "other",
        "gold_should_escalate": True,
        "gold_reason_notes": "B2B liquidation inquiry outside standard retail customer support.",
        "reference_reply": "For wholesale and bulk inventory liquidation, please visit Amazon Bulk Liquidations portal."
    },
    {
        "customer_msg": "I am reporting your company to the Federal Trade Commission and state Attorney General for fraud.",
        "gold_intent": "other",
        "gold_should_escalate": True,
        "gold_reason_notes": "Regulatory complaint / government agency escalation threat.",
        "reference_reply": "We take regulatory concerns seriously. Please contact our corporate resolutions team: [link]."
    },
    {
        "customer_msg": "Does Amazon Web Services offer student cloud credits for university machine learning coursework?",
        "gold_intent": "other",
        "gold_should_escalate": True,
        "gold_reason_notes": "AWS cloud computing inquiry rather than retail e-commerce.",
        "reference_reply": "For AWS student credits and cloud support, please visit the AWS Educate portal."
    },
    {
        "customer_msg": "Can you tell me what the current stock price of AMZN is on NASDAQ right now?",
        "gold_intent": "other",
        "gold_should_escalate": True,
        "gold_reason_notes": "Financial markets query out of customer support scope.",
        "reference_reply": "For investor relations and stock quotes, please visit ir.aboutamazon.com."
    },
    {
        "customer_msg": "How can our charity organization register to receive donations through AmazonSmile?",
        "gold_intent": "other",
        "gold_should_escalate": True,
        "gold_reason_notes": "Charity / AmazonSmile program inquiry.",
        "reference_reply": "Please visit org.amazon.com for information regarding non-profit and charitable giving programs."
    }
]

def main():
    out_path = Path("golden_set/golden_150.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Total golden items to write: {len(GOLDEN_ITEMS)}")
    assert len(GOLDEN_ITEMS) == 150, f"Expected exactly 150 items, got {len(GOLDEN_ITEMS)}"
    
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, item in enumerate(GOLDEN_ITEMS, 1):
            record = {
                "id": idx,
                "customer_msg": item["customer_msg"],
                "gold_intent": item["gold_intent"],
                "gold_should_escalate": item["gold_should_escalate"],
                "gold_reason_notes": item["gold_reason_notes"],
                "reference_reply": item["reference_reply"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print(f"Successfully generated {len(GOLDEN_ITEMS)} golden items in {out_path}")

if __name__ == "__main__":
    main()
