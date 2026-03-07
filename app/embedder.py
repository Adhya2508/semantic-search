"""
Embedding helper for semantic retrieval.

This module loads a SentenceTransformer model and exposes a single encode()
function used throughout the system.

Design decisions:

1. Singleton pattern
   Embedding models are large (~400MB for MPNet). Loading multiple instances
   would waste memory and significantly slow the system. A module-level
   singleton ensures the model is loaded exactly once.

2. Normalised embeddings
   Sentence-transformer models are trained using cosine similarity. We
   therefore L2-normalise vectors so that cosine similarity becomes equivalent
   to inner product. This allows FAISS to use a fast IndexFlatIP index.

3. Float32 precision
   FAISS expects float32 vectors. Converting here avoids repeated casting
   during indexing and search.
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config import MODEL_PATH, EMBEDDING_DIM


class Embedder:

    _instance = None

    def __new__(cls):
        """
        Lazy-load singleton model instance.
        """

        if cls._instance is None:

            cls._instance = super().__new__(cls)

            # Choose device automatically
            device = "cuda" if torch.cuda.is_available() else "cpu"

            cls._instance._model = SentenceTransformer(
                MODEL_PATH,
                device=device
            )

            print(f"[Embedder] Loaded model from {MODEL_PATH} on {device}")

        return cls._instance

    def encode(self, texts: list[str] | str, batch_size: int = 256) -> np.ndarray:
        """
        Encode text into semantic embeddings.

        Parameters
        ----------
        texts : list[str] or str
            Input documents or query.

        batch_size : int
            Batch size for model inference. Larger batches improve throughput
            but increase memory usage.

        Returns
        -------
        np.ndarray
            L2-normalised float32 embeddings.

            Shape:
                (N, EMBEDDING_DIM) for multiple inputs
                (EMBEDDING_DIM,) for a single string
        """

        single_input = isinstance(texts, str)

        if single_input:
            texts = [texts]

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 1000
        )

        embeddings = embeddings.astype(np.float32)

        # Safety check – prevents subtle FAISS bugs if wrong model is used
        if embeddings.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, "
                f"got {embeddings.shape[1]}"
            )

        return embeddings[0] if single_input else embeddings