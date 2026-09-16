"""
src/retrieval/retriever.py

Semantic RAG retriever combining SentenceTransformers with FAISS vector store.
Provides grounded evidence from historical resolved customer conversations.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import FAISSVectorStore

class SupportCaseRetriever:
    def __init__(
        self,
        index_path: str = "data/processed/faiss_index.bin",
        metadata_path: str = "data/processed/vector_metadata.json",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedder = EmbeddingModel(model_name=embedding_model_name)
        self.vector_store = FAISSVectorStore(dim=self.embedder.embedding_dim)
        self._is_loaded = False

        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.load()

    def load(self) -> None:
        """Load vector store from disk."""
        self.vector_store.load(self.index_path, self.metadata_path)
        self._is_loaded = True

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        intent_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top_k historical cases similar to query.
        Returns list of evidence dicts with conversation_id, customer_msg, agent_reply, similarity.
        """
        if not self._is_loaded:
            raise RuntimeError(
                f"Retriever index not loaded. Run 'python scripts/build_index.py' first."
            )

        query_vec = self.embedder.encode([query])[0]
        # Fetch slightly more if filtering by intent
        fetch_k = top_k * 3 if intent_filter else top_k
        raw_matches = self.vector_store.search(query_vec, top_k=fetch_k)

        evidences = []
        for meta, sim in raw_matches:
            if intent_filter and meta.get("intent") and meta.get("intent") != intent_filter:
                continue

            evidences.append({
                "conversation_id": str(meta.get("conversation_id", "")),
                "customer_msg": meta.get("customer_msg", ""),
                "agent_reply": meta.get("agent_reply", ""),
                "similarity": round(float(sim), 4),
                "intent": meta.get("intent", "other")
            })

            if len(evidences) >= top_k:
                break

        return evidences
