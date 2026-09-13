"""
src/intent_classifier.py

Hybrid Intent Classifier:
1. Few-shot LLM classifier using gemini-3.6-flash with in-context examples.
2. Classical ML baseline (TF-IDF + Logistic Regression) for latency/accuracy benchmarking.
"""

import os
import sys
import re
import json
import argparse
from typing import Dict, Any, List
from pathlib import Path

# Ensure repo root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.config import (
    INTENTS,
    INTENTS_LABELED_PATH,
    LLM_MODEL,
    get_llm_client
)

def load_seed_examples() -> List[Dict[str, str]]:
    """Loads seed labeled examples from intents_labeled.jsonl."""
    examples = []
    if os.path.exists(INTENTS_LABELED_PATH):
        with open(INTENTS_LABELED_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line.strip()))
    return examples

class FewShotLLMClassifier:
    """Few-shot in-context classifier using LLM."""
    
    def __init__(self, model_name: str = None, provider: str = None):
        self.model_name = model_name or LLM_MODEL
        self.client = get_llm_client(provider=provider)
        self.seed_examples = load_seed_examples()
        self._build_system_prompt()
        
    def _build_system_prompt(self):
        intent_descriptions = "\n".join([f"- {intent}" for intent in INTENTS])
        
        # Select 3 distinct examples per intent
        examples_by_intent = {}
        for ex in self.seed_examples:
            intent = ex["intent"]
            examples_by_intent.setdefault(intent, []).append(ex["text"])
            
        examples_block = ""
        for intent in INTENTS:
            examples = examples_by_intent.get(intent, [])[:3]
            examples_block += f"\nIntent: {intent}\n"
            for ex in examples:
                examples_block += f"  Example: \"{ex}\"\n"
                
        self.system_prompt = (
            "You are an expert customer support intent classifier for Amazon customer service.\n"
            "Your task is to classify an incoming customer message into EXACTLY ONE of the following intents:\n"
            f"{intent_descriptions}\n\n"
            "Canonical Reference Examples:\n"
            f"{examples_block}\n\n"
            "Instructions:\n"
            "1. Output valid JSON ONLY with the following schema:\n"
            "   {\n"
            "     \"intent\": \"<intent_name>\",\n"
            "     \"confidence\": <float between 0.0 and 1.0>,\n"
            "     \"reasoning\": \"<brief 1-sentence justification>\"\n"
            "   }\n"
            "2. If the message is ambiguous, out-of-scope, non-English, or expresses legal action, classify it as 'other'.\n"
            "3. Do not include markdown formatting or commentary outside the JSON."
        )

    def classify(self, message: str) -> Dict[str, Any]:
        """Classifies a customer message into an intent with confidence."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Customer Message: {message}"}
                ]
            )
            raw_content = response.choices[0].message.content.strip()
            
            # Clean possible markdown fencing
            clean_json = raw_content
            if clean_json.startswith("```"):
                clean_json = re.sub(r"^```(?:json)?\s*", "", clean_json)
                clean_json = re.sub(r"\s*```$", "", clean_json)
                
            data = json.loads(clean_json)
            predicted_intent = data.get("intent", "other")
            if predicted_intent not in INTENTS:
                predicted_intent = "other"
                
            confidence = float(data.get("confidence", 0.7))
            reasoning = data.get("reasoning", "")
            
            return {
                "intent": predicted_intent,
                "confidence": round(confidence, 3),
                "reasoning": reasoning
            }
        except Exception as e:
            # Safe fallback if API error or malformed JSON
            return {
                "intent": "other",
                "confidence": 0.5,
                "reasoning": f"Classifier fallback due to error: {str(e)}"
            }

class BaselineTfidfClassifier:
    """Classical ML baseline classifier using TF-IDF + Logistic Regression."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=3000)
        self.model = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
        self.is_trained = False
        self._train()
        
    def _train(self):
        examples = load_seed_examples()
        if not examples:
            return
        X_texts = [ex["text"] for ex in examples]
        y_labels = [ex["intent"] for ex in examples]
        
        X_vec = self.vectorizer.fit_transform(X_texts)
        self.model.fit(X_vec, y_labels)
        self.is_trained = True
        
    def classify(self, message: str) -> Dict[str, Any]:
        if not self.is_trained:
            return {"intent": "other", "confidence": 0.5, "reasoning": "Baseline model not trained."}
            
        X_vec = self.vectorizer.transform([message])
        probs = self.model.predict_proba(X_vec)[0]
        classes = self.model.classes_
        
        top_idx = np.argmax(probs)
        predicted_intent = classes[top_idx]
        confidence = float(probs[top_idx])
        
        return {
            "intent": predicted_intent,
            "confidence": round(confidence, 3),
            "reasoning": f"TF-IDF LogisticRegression argmax probability {confidence:.2f}"
        }

def get_intent_classifier(use_baseline: bool = False, model_name: str = None, provider: str = None):
    """Factory to retrieve either the LLM classifier or classical baseline."""
    if use_baseline:
        return BaselineTfidfClassifier()
    return FewShotLLMClassifier(model_name=model_name, provider=provider)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", type=str, default="Where is my parcel? It was supposed to be here yesterday.")
    parser.add_argument("--baseline", action="store_true", help="Use TF-IDF baseline instead of LLM")
    args = parser.parse_args()
    
    classifier = get_intent_classifier(use_baseline=args.baseline)
    result = classifier.classify(args.message)
    print(f"\nModel: {'Baseline TF-IDF' if args.baseline else 'Few-Shot LLM'}")
    print(f"Message: {args.message}")
    print(f"Result: {json.dumps(result, indent=2)}\n")
