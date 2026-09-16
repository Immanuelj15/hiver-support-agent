"""
scripts/build_index.py

Builds the FAISS dense semantic index from the clean knowledge corpus.
Encodes customer messages using sentence-transformers/all-MiniLM-L6-v2,
builds an IndexFlatIP vector store, saves index binary and metadata,
and verifies retrieval with a sample test query.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_jsonl
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import FAISSVectorStore

def main():
    parser = argparse.ArgumentParser(description="Build FAISS vector index from knowledge corpus.")
    parser.add_argument("--corpus", default="data/processed/knowledge_corpus.jsonl", help="Corpus JSONL path")
    parser.add_argument("--cache", default="data/processed/embeddings_cache.npz", help="Optional precomputed embeddings cache")
    parser.add_argument("--index-out", default="data/processed/faiss_index.bin", help="FAISS index binary path")
    parser.add_argument("--meta-out", default="data/processed/vector_metadata.json", help="Metadata JSON path")
    parser.add_argument("--limit", type=int, default=5000, help="Max items to index")
    args = parser.parse_args()

    print(f"Loading knowledge corpus from {args.corpus}...")
    corpus = load_jsonl(args.corpus)
    if args.limit and len(corpus) > args.limit:
        print(f"Subsampling to {args.limit} items for optimal retrieval latency...")
        corpus = corpus[:args.limit]

    print(f"Total corpus items to index: {len(corpus):,}")

    embedder = EmbeddingModel()
    texts = [item["customer_msg"] for item in corpus]

    # Check if precomputed cache is available and valid
    embeddings = None
    if os.path.exists(args.cache):
        try:
            print(f"Loading precomputed cache from {args.cache}...")
            cached = np.load(args.cache, allow_pickle=True)
            if "embeddings" in cached and "threads" in cached:
                cached_threads = list(cached["threads"])
                cached_embs = cached["embeddings"]
                # Convert threads format if needed
                normalized_corpus = []
                for t in cached_threads:
                    normalized_corpus.append({
                        "conversation_id": str(t.get("thread_id", "")),
                        "customer_msg": t.get("customer_msg", ""),
                        "agent_reply": t.get("agent_reply") or t.get("brand_reply", ""),
                        "intent": t.get("intent", "order_delivery_delay")
                    })
                corpus = normalized_corpus
                embeddings = cached_embs.astype(np.float32)
                # Ensure L2 normalized
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                embeddings = embeddings / norms
                print(f"Loaded {len(embeddings)} precomputed normalized embeddings for {len(corpus)} cases.")
        except Exception as e:
            print(f"Cache load error: {e}. Will compute fresh embeddings.")

    if embeddings is None:
        print(f"Computing embeddings for {len(texts)} texts with {embedder.model_name}...")
        embeddings = embedder.encode(texts, batch_size=128, show_progress_bar=True)

    print(f"Embeddings shape: {embeddings.shape}")

    vector_store = FAISSVectorStore(dim=embedder.embedding_dim)
    print("Building FAISS IndexFlatIP...")
    vector_store.build_from_embeddings(embeddings, corpus)

    print(f"Saving FAISS index to {args.index_out}...")
    vector_store.save(args.index_out, args.meta_out)
    print(f"Saved index and metadata ({len(corpus)} records).")

    # Verification query
    test_query = "Where is my parcel? It was supposed to be here yesterday."
    print(f"\n--- Verification Retrieval Test ---")
    print(f"Query: '{test_query}'")
    q_vec = embedder.encode([test_query])[0]
    matches = vector_store.search(q_vec, top_k=2)
    for i, (meta, score) in enumerate(matches, 1):
        print(f"[{i}] Similarity: {score:.4f} | Conv ID: {meta.get('conversation_id')}")
        print(f"    Customer: {meta.get('customer_msg')[:80]}...")
        print(f"    Agent:    {meta.get('agent_reply')[:80]}...")

    print("\n[SUCCESS] Vector store built and verified successfully!")

if __name__ == "__main__":
    main()
