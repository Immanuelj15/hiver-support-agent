"""
src/retriever.py

Retrieval-Augmented Generation (RAG) Index:
- Builds sentence-transformers embedding index over historical resolved customer threads.
- Buckets resolutions by intent for high-precision semantic search.
- Computes cosine similarity and retrieves top-k most relevant historical (query, reply) pairs.
- Caches index to disk for fast startup.
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Ensure repo root in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import (
    THREADS_PATH,
    EMBEDDINGS_CACHE_PATH,
    EMBEDDING_MODEL_NAME,
    TOP_K_RETRIEVAL,
    INTENTS
)
from src.intent_classifier import BaselineTfidfClassifier

class HistoricalRetriever:
    """Retrieves top-k historically resolved support threads filtered by intent."""
    
    def __init__(
        self,
        threads_path: str = str(THREADS_PATH),
        cache_path: str = str(EMBEDDINGS_CACHE_PATH),
        model_name: str = EMBEDDING_MODEL_NAME,
        max_threads_to_index: int = 4000
    ):
        self.threads_path = threads_path
        self.cache_path = cache_path
        self.model_name = model_name
        self.max_threads_to_index = max_threads_to_index
        
        print(f"Loading embedding model '{model_name}'...")
        self.encoder = SentenceTransformer(model_name)
        
        self.threads: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray = np.empty((0, 384))
        self.intents: List[str] = []
        
        self._load_or_build_index()
        
    def _load_or_build_index(self):
        """Loads cached index if available, otherwise builds from threads.jsonl."""
        if os.path.exists(self.cache_path):
            print(f"Loading cached embeddings from {self.cache_path}...")
            data = np.load(self.cache_path, allow_pickle=True)
            self.embeddings = data["embeddings"]
            self.intents = list(data["intents"])
            self.threads = list(data["threads"])
            print(f"Retriever ready with {len(self.threads):,} cached historical cases.")
            return
            
        print(f"Building retrieval index from {self.threads_path} (limit={self.max_threads_to_index})...")
        if not os.path.exists(self.threads_path):
            raise FileNotFoundError(f"Threads file not found: {self.threads_path}. Run src/ingest.py first.")
            
        raw_threads = []
        with open(self.threads_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw_threads.append(json.loads(line.strip()))
                    if len(raw_threads) >= self.max_threads_to_index:
                        break
                        
        print(f"Assigning intent buckets to {len(raw_threads):,} historical threads...")
        classifier = BaselineTfidfClassifier()
        
        valid_threads = []
        intents_list = []
        customer_msgs = []
        
        for t in raw_threads:
            cust_msg = t.get("customer_msg", "").strip()
            brand_rep = t.get("brand_reply", "").strip()
            if len(cust_msg.split()) >= 4 and len(brand_rep.split()) >= 4:
                pred = classifier.classify(cust_msg)
                assigned_intent = pred["intent"]
                valid_threads.append({
                    "thread_id": t["thread_id"],
                    "customer_msg": cust_msg,
                    "brand_reply": brand_rep,
                    "intent": assigned_intent
                })
                intents_list.append(assigned_intent)
                customer_msgs.append(cust_msg)
                
        print(f"Computing embeddings for {len(customer_msgs):,} customer messages...")
        embeddings = self.encoder.encode(
            customer_msgs,
            show_progress_bar=True,
            normalize_embeddings=True,
            batch_size=64
        )
        
        self.embeddings = np.array(embeddings, dtype=np.float32)
        self.intents = intents_list
        self.threads = valid_threads
        
        # Save cache
        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.cache_path,
            embeddings=self.embeddings,
            intents=np.array(self.intents),
            threads=np.array(self.threads, dtype=object)
        )
        print(f"Saved retrieval index cache to {self.cache_path} ({len(self.threads):,} records).")

    def retrieve(
        self,
        query: str,
        intent: str = None,
        k: int = TOP_K_RETRIEVAL
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Retrieves top-k historical pairs from the same intent bucket (or global fallback).
        Returns: (list_of_pairs, max_similarity_score)
        """
        if len(self.threads) == 0:
            return [], 0.0
            
        # Encode query (normalized vector)
        q_emb = self.encoder.encode([query], normalize_embeddings=True)[0]
        
        # Filter indices by intent if provided
        target_indices = []
        if intent and intent in INTENTS:
            target_indices = [i for i, item_intent in enumerate(self.intents) if item_intent == intent]
            
        # Fallback to all indices if target intent bucket is too small (< k)
        if len(target_indices) < k:
            candidate_indices = np.arange(len(self.threads))
        else:
            candidate_indices = np.array(target_indices)
            
        candidate_embeds = self.embeddings[candidate_indices]
        # Dot product of normalized vectors = cosine similarity
        sims = np.dot(candidate_embeds, q_emb)
        
        # Top-k largest similarities
        top_k_ranks = np.argsort(-sims)[:k]
        
        results = []
        max_sim = 0.0
        for rank in top_k_ranks:
            orig_idx = candidate_indices[rank]
            score = float(sims[rank])
            if score > max_sim:
                max_sim = score
            t = self.threads[orig_idx]
            results.append({
                "thread_id": t["thread_id"],
                "customer_msg": t["customer_msg"],
                "brand_reply": t["brand_reply"],
                "intent": t.get("intent", ""),
                "similarity": round(score, 4)
            })
            
        return results, round(max_sim, 4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="Where is my parcel? Delivered status but not in mailbox.")
    parser.add_argument("--intent", type=str, default="order_delivery_delay")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    
    retriever = HistoricalRetriever()
    results, max_sim = retriever.retrieve(args.query, intent=args.intent, k=args.k)
    print(f"\nQuery: {args.query}")
    print(f"Target Intent: {args.intent}")
    print(f"Max Similarity: {max_sim}")
    print("\nTop Retrieved Historical Cases:")
    for i, res in enumerate(results, 1):
        print(f"\n[{i}] Similarity: {res['similarity']} (Intent: {res['intent']})")
        print(f"    Customer: {res['customer_msg']}")
        print(f"    Amazon:   {res['brand_reply']}")
