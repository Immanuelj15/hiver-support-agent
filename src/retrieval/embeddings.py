"""
src/retrieval/embeddings.py

Wrapper for SentenceTransformer embedding models.
Uses all-MiniLM-L6-v2 by default with L2 normalization for cosine similarity search.
"""

import os
from typing import List, Union
import numpy as np

# Prefer local cached models to avoid DNS connection timeouts
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        try:
            self._model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            # If local_files_only strictly fails, allow online lookup
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            self._model = SentenceTransformer(model_name)



    @property
    def embedding_dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 64,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Encode text(s) into normalized float32 embeddings.
        With normalization, inner product equals cosine similarity.
        """
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)
