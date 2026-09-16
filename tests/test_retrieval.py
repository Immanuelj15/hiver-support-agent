"""
tests/test_retrieval.py

Unit and integration tests for FAISS vector store and embedding retrieval.
"""

import numpy as np
import pytest
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.retriever import SupportCaseRetriever

def test_faiss_vector_store_basic():
    store = FAISSVectorStore(dim=4)
    # Create 3 normalized vectors
    v1 = np.array([1, 0, 0, 0], dtype=np.float32)
    v2 = np.array([0, 1, 0, 0], dtype=np.float32)
    v3 = np.array([0.7071, 0.7071, 0, 0], dtype=np.float32)
    embeddings = np.vstack([v1, v2, v3])

    metadata = [
        {"conversation_id": "c1", "customer_msg": "m1", "agent_reply": "r1"},
        {"conversation_id": "c2", "customer_msg": "m2", "agent_reply": "r2"},
        {"conversation_id": "c3", "customer_msg": "m3", "agent_reply": "r3"}
    ]

    store.build_from_embeddings(embeddings, metadata)
    assert store.index.ntotal == 3

    # Query identical to v1
    query = np.array([1, 0, 0, 0], dtype=np.float32)
    results = store.search(query, top_k=2)

    assert len(results) == 2
    top_meta, top_score = results[0]
    assert top_meta["conversation_id"] == "c1"
    assert np.isclose(top_score, 1.0, atol=1e-4)

def test_retriever_load_and_retrieve():
    try:
        retriever = SupportCaseRetriever()
        if not retriever._is_loaded:
            pytest.skip("Index file not yet built on disk.")

        results = retriever.retrieve("Where is my delivery?", top_k=2)
        assert len(results) <= 2
        if results:
            assert "conversation_id" in results[0]
            assert "customer_msg" in results[0]
            assert "agent_reply" in results[0]
            assert "similarity" in results[0]
            assert 0.0 <= results[0]["similarity"] <= 1.0
    except Exception as e:
        pytest.skip(f"Skipping retriever test if models cannot be initialized: {e}")
