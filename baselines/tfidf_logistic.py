"""
baselines/tfidf_logistic.py

Baseline 2: TF-IDF Vectorizer + Multinomial Logistic Regression Classifier.
Trained on disjoint training samples across all 8 intents.
Evaluates against the golden benchmark, outputs results to results/baseline_results.json,
and generates results/figures/confusion_matrix.png.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.intents.taxonomy import ALL_INTENTS
from src.data.loader import load_jsonl

def build_training_dataset(labeled_path: str, threads_path: str, golden_msgs: set) -> tuple:
    """Builds a balanced, leakage-free training dataset for TF-IDF training."""
    X_train = []
    y_train = []

    # 1. Load labeled seed examples
    if os.path.exists(labeled_path):
        seeds = load_jsonl(labeled_path)
        for s in seeds:
            text = s.get("text") or s.get("customer_msg", "")
            intent = s.get("intent") or s.get("gold_intent", "")
            if text and intent in ALL_INTENTS and text.strip().lower() not in golden_msgs:
                X_train.append(text)
                y_train.append(intent)

    # 2. Add high-confidence heuristic examples from historical threads (excluding golden set)
    if os.path.exists(threads_path):
        threads = load_jsonl(threads_path)
        intent_quotas = {intent: 60 for intent in ALL_INTENTS}
        # Account for existing seeds
        for y in y_train:
            if y in intent_quotas:
                intent_quotas[y] -= 1

        for item in threads:
            msg = item.get("customer_msg", "").strip()
            if not msg or msg.lower() in golden_msgs or len(msg) < 15:
                continue

            lower = msg.lower()
            detected = None
            if any(w in lower for w in ["hacked", "stolen", "unauthorized", "otp", "2fa", "fraud"]) and intent_quotas["account_security_and_login"] > 0:
                detected = "account_security_and_login"
            elif any(w in lower for w in ["shattered", "broken", "leaking", "wrong item", "damaged box"]) and intent_quotas["damaged_or_wrong_item"] > 0:
                detected = "damaged_or_wrong_item"
            elif any(w in lower for w in ["return label", "drop off", "refund status", "whole foods return"]) and intent_quotas["return_and_refund"] > 0:
                detected = "return_and_refund"
            elif any(w in lower for w in ["prime renewal", "subscription", "annual fee", "double charged", "billed twice"]) and intent_quotas["subscription_and_billing"] > 0:
                detected = "subscription_and_billing"
            elif any(w in lower for w in ["kindle", "fire stick", "echo dot", "alexa app", "won't turn on", "black screen"]) and intent_quotas["product_technical_issue"] > 0:
                detected = "product_technical_issue"
            elif any(w in lower for w in ["terrible service", "driver threw", "rude agent", "hung up on me"]) and intent_quotas["feedback_or_complaint"] > 0:
                detected = "feedback_or_complaint"
            elif any(w in lower for w in ["where is my parcel", "delivery delayed", "package late", "delivered Saturday"]) and intent_quotas["order_delivery_delay"] > 0:
                detected = "order_delivery_delay"
            elif any(w in lower for w in ["good morning", "hello there", "walmart price", "thank you"]) and intent_quotas["other"] > 0:
                detected = "other"

            if detected:
                X_train.append(msg)
                y_train.append(detected)
                intent_quotas[detected] -= 1

            if all(v <= 0 for v in intent_quotas.values()):
                break

    return X_train, y_train

def plot_confusion_matrix(cm, classes, output_path: str):
    """Plot and save confusion matrix visualization."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title="Confusion Matrix: TF-IDF + Logistic Regression Baseline",
        ylabel="True Golden Intent",
        xlabel="Predicted Intent"
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved to {output_path}")

def run_tfidf_baseline(
    golden_path: str = "data/golden/golden_set.jsonl",
    labeled_path: str = "data/processed/intents_labeled.jsonl",
    threads_path: str = "data/processed/threads.jsonl",
    results_path: str = "results/baseline_results.json",
    fig_path: str = "results/figures/confusion_matrix.png"
):
    print("Building training dataset for TF-IDF baseline...")
    golden_items = load_jsonl(golden_path)
    golden_msgs = {item["customer_msg"].strip().lower() for item in golden_items}

    X_train, y_train = build_training_dataset(labeled_path, threads_path, golden_msgs)
    print(f"Training dataset: {len(X_train)} samples across {len(set(y_train))} classes.")

    X_test = [item["customer_msg"] for item in golden_items]
    y_test = [item["gold_intent"] for item in golden_items]

    # Pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ])

    print("Fitting TF-IDF + Logistic Regression model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating against golden evaluation benchmark...")
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    labels = sorted(list(set(y_test) | set(y_pred)))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plot_confusion_matrix(cm, labels, fig_path)

    report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)

    results = {
        "baseline_model": "TF-IDF (1-2 gram) + Logistic Regression (Balanced)",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_w), 4),
        "per_class_metrics": {
            k: {
                "precision": round(v["precision"], 4),
                "recall": round(v["recall"], 4),
                "f1": round(v["f1-score"], 4),
                "support": v["support"]
            }
            for k, v in report.items() if k in ALL_INTENTS
        }
    }

    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 65)
    print("BASELINE 2: TF-IDF + LOGISTIC REGRESSION EVALUATION")
    print("=" * 65)
    print(f"Accuracy:           {results['accuracy'] * 100:.2f}%")
    print(f"Macro Precision:    {results['macro_precision']:.4f}")
    print(f"Macro Recall:       {results['macro_recall']:.4f}")
    print(f"Macro F1 Score:     {results['macro_f1']:.4f}")
    print(f"Weighted F1:        {results['weighted_f1']:.4f}")
    print(f"Results saved to:   {results_path}")
    print("=" * 65)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="data/golden/golden_set.jsonl")
    parser.add_argument("--output", default="results/baseline_results.json")
    args = parser.parse_args()
    run_tfidf_baseline(golden_path=args.golden, results_path=args.output)
