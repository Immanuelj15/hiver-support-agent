"""
src/retrieval/vector_store.py

FAISS vector store manager for dense semantic similarity search.
Uses IndexFlatIP on normalized vectors for exact cosine similarity.
Saves index to binary file and associated metadata to JSON.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import faiss

class FAISSVectorStore:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict[str, Any]] = []

    def build_from_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]]
    ) -> None:
        """Initialize index from matrix of embeddings and parallel metadata."""
        assert len(embeddings) == len(metadata), "Embeddings and metadata count must match"
        self.dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings.astype(np.float32))
        self.metadata = list(metadata)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 3
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for top_k nearest neighbors by cosine similarity.
        query_vector should be shape (dim,) or (1, dim).
        Returns list of (metadata_dict, similarity_score).
        """
        if self.index is None:
            raise RuntimeError("Index is not loaded or built.")

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        scores, indices = self.index.search(query_vector.astype(np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append((self.metadata[idx], float(score)))

        return results

    def save(self, index_path: str, metadata_path: str) -> None:
        """Persist FAISS index binary and metadata JSON to disk."""
        if self.index is None:
            raise RuntimeError("No index to save.")
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, index_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)

    def load(self, index_path: str, metadata_path: str) -> None:
        """Load FAISS index binary and metadata JSON from disk."""
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Missing index or metadata: {index_path}, {metadata_path}")

        self.index = faiss.read_index(index_path)
        self.dim = self.index.d
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
