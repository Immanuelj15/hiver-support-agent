"""
notebooks/cluster_intents.py

Embeds ~500 sampled customer messages using sentence-transformers (all-MiniLM-L6-v2)
and clusters them with k-Means (k=8). Prints representative customer messages per cluster
to discover and empirically validate an intent taxonomy.
"""

import json
import random
import argparse
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

def cluster_intents(
    threads_path: str = "data/processed/threads.jsonl",
    n_samples: int = 500,
    k: int = 8,
    seed: int = 42
):
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"Reading messages from {threads_path}...")
    messages = []
    with open(threads_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            msg = data.get("customer_msg", "").strip()
            if len(msg.split()) >= 6: # sufficiently detailed
                messages.append(msg)
                
    print(f"Loaded {len(messages):,} candidate messages.")
    if len(messages) > n_samples:
        sampled_messages = random.sample(messages, n_samples)
    else:
        sampled_messages = messages
        
    print(f"Embedding {len(sampled_messages)} messages with all-MiniLM-L6-v2...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(sampled_messages, show_progress_bar=True, normalize_embeddings=True)
    
    print(f"Running k-Means with k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Find examples closest to centroids
    print("\n" + "="*80)
    print(f"DISCOVERED INTENT CLUSTERS (k={k})")
    print("="*80)
    
    cluster_data = {}
    for cluster_id in range(k):
        centroid = kmeans.cluster_centers_[cluster_id]
        indices = np.where(cluster_labels == cluster_id)[0]
        
        # Compute cosine similarity to centroid (since embeddings are normalized, dot product = cosine sim)
        cluster_embeds = embeddings[indices]
        sims = np.dot(cluster_embeds, centroid)
        sorted_indices = indices[np.argsort(-sims)]
        
        print(f"\n[CLUSTER {cluster_id}] (Size: {len(indices)} messages)")
        top_examples = []
        for rank, idx in enumerate(sorted_indices[:12]):
            ex = sampled_messages[idx]
            top_examples.append(ex)
            # Safe print for Windows consoles
            safe_text = ex.encode("ascii", "replace").decode("ascii")
            print(f"  {rank+1:2d}. {safe_text}")
            
        cluster_data[f"cluster_{cluster_id}"] = {
            "size": int(len(indices)),
            "top_examples": top_examples
        }
        
    # Save output for reference
    out_file = Path("data/processed/cluster_discovery.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(cluster_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved cluster discovery analysis to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads_path", type=str, default="data/processed/threads.jsonl")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()
    
    cluster_intents(
        threads_path=args.threads_path,
        n_samples=args.n_samples,
        k=args.k
    )
