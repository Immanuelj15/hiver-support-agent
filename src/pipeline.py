"""
src/pipeline.py

End-to-End Inference Pipeline:
Wires together Intent Classification -> Semantic RAG Retrieval -> Grounded Reply Drafting -> Deterministic Escalation.
Public interface: run(customer_msg) -> Dict[str, Any]
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any
from pathlib import Path

# Ensure repo root in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    DEFAULT_SIMILARITY_THRESHOLD,
    TOP_K_RETRIEVAL
)
from src.intent_classifier import get_intent_classifier
from src.retriever import HistoricalRetriever
from src.reply_agent import ReplyAgent
from src.escalation import EscalationPolicy

class SupportAgentPipeline:
    """Orchestrates intent classification, retrieval, reply generation, and escalation routing."""
    
    def __init__(
        self,
        use_baseline_classifier: bool = False,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        top_k: int = TOP_K_RETRIEVAL,
        model_name: str = None,
        provider: str = None
    ):
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.use_baseline_classifier = use_baseline_classifier
        self.provider = provider
        self.model_name = model_name
        
        print(f"Initializing SupportAgentPipeline (Provider: {provider or 'default'}, Model: {model_name or 'default'})...")
        self.classifier = get_intent_classifier(
            use_baseline=use_baseline_classifier,
            model_name=model_name,
            provider=provider
        )
        self.retriever = HistoricalRetriever()
        self.reply_agent = ReplyAgent(model_name=model_name, provider=provider)
        self.escalation_policy = EscalationPolicy(similarity_threshold=similarity_threshold)
        print("SupportAgentPipeline initialized successfully.")

    def run(self, customer_msg: str) -> Dict[str, Any]:
        """
        Executes complete pipeline for a single customer message.
        """
        start_time = time.time()
        
        # 1. Intent Classification
        intent_res = self.classifier.classify(customer_msg)
        intent = intent_res["intent"]
        confidence = intent_res["confidence"]
        intent_reasoning = intent_res.get("reasoning", "")
        
        # 2. Historical Resolution Retrieval
        retrieved_pairs, max_sim = self.retriever.retrieve(
            query=customer_msg,
            intent=intent,
            k=self.top_k
        )
        
        # 3. Grounded Reply Drafting
        reply_res = self.reply_agent.draft_reply(
            customer_msg=customer_msg,
            intent=intent,
            retrieved_pairs=retrieved_pairs,
            max_similarity=max_sim
        )
        draft_reply = reply_res["draft_reply"]
        grounding_note = reply_res["grounding_note"]
        
        # 4. Multi-Criteria Escalation Decision
        escalation_res = self.escalation_policy.evaluate(
            customer_msg=customer_msg,
            intent=intent,
            confidence=confidence,
            max_similarity=max_sim,
            reply_draft=draft_reply
        )
        decision = escalation_res["decision"]
        reason = escalation_res["reason"]
        
        total_latency_ms = round((time.time() - start_time) * 1000, 1)
        
        return {
            "customer_msg": customer_msg,
            "intent": intent,
            "confidence": confidence,
            "intent_reasoning": intent_reasoning,
            "draft_reply": draft_reply,
            "grounding_note": grounding_note,
            "decision": decision,
            "reason": reason,
            "max_similarity": max_sim,
            "retrieved_pairs": retrieved_pairs,
            "latency_ms": total_latency_ms
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Support Agent Pipeline on single or multiple messages.")
    parser.add_argument("--message", type=str, default=None, help="Single customer message to test")
    parser.add_argument("--file", "-f", type=str, default=None, help="Path to text file with one customer message per line")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive prompt mode to type multiple messages")
    parser.add_argument("--baseline", action="store_true", help="Use TF-IDF classifier instead of LLM")
    parser.add_argument("--threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD, help="Similarity threshold for escalation")
    parser.add_argument("--ollama", action="store_true", help="Use local Ollama model")
    parser.add_argument("--model", type=str, default=None, help="Custom model name (e.g. mistral:latest, phi3:latest)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Optional output JSON file for batch results")
    args = parser.parse_args()
    
    provider = "ollama" if args.ollama else None
    model_name = args.model or ("mistral:latest" if args.ollama else None)
    
    pipeline = SupportAgentPipeline(
        use_baseline_classifier=args.baseline,
        similarity_threshold=args.threshold,
        model_name=model_name,
        provider=provider
    )
    
    def print_result(res: Dict[str, Any], idx: int = None):
        prefix = f"[{idx}] " if idx is not None else ""
        print("\n" + "="*80)
        print(f"PIPELINE EXECUTION RESULT {prefix}")
        print("="*80)
        print(f"Customer Message:  {res['customer_msg']}")
        print(f"Predicted Intent:  {res['intent']} (confidence: {res['confidence']})")
        print(f"Retrieval Sim:     {res['max_similarity']}")
        print(f"Draft Reply:       {res['draft_reply']}")
        print(f"Grounding Note:    {res['grounding_note']}")
        print(f"Escalation Action: {res['decision'].upper()}")
        print(f"Decision Reason:   {res['reason']}")
        print(f"Total Latency:     {res['latency_ms']} ms")
        print("="*80)

    # 1. Interactive Loop Mode
    if args.interactive:
        print("\n" + "#"*80)
        print("INTERACTIVE SUPPORT AGENT MODE")
        print("Type any customer inquiry and press Enter. Type 'exit' or 'quit' to stop.")
        print("#"*80 + "\n")
        
        count = 1
        while True:
            try:
                user_query = input(f"\n[Inquiry {count}] Enter customer message: ").strip()
                if not user_query:
                    continue
                if user_query.lower() in ("exit", "quit", "q"):
                    print("Exiting interactive mode.")
                    break
                res = pipeline.run(user_query)
                print_result(res, count)
                count += 1
            except (KeyboardInterrupt, EOFError):
                print("\nExiting interactive mode.")
                break

    # 2. File Batch Mode (multiple texts)
    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File '{args.file}' not found.")
            sys.exit(1)
            
        messages = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    messages.append(stripped)
                    
        print(f"\nProcessing {len(messages)} messages from '{args.file}'...\n")
        all_results = []
        for idx, msg in enumerate(messages, 1):
            res = pipeline.run(msg)
            all_results.append(res)
            print_result(res, idx)
            
        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"\nSaved all {len(all_results)} results to {out_p}")
            
    # 3. Single Message Mode
    else:
        query = args.message or "I received a shattered coffee mug, can I get a replacement?"
        res = pipeline.run(query)
        print_result(res)

