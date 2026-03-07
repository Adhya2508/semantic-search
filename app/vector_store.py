"""
Vector store built on top of FAISS.

We use IndexFlatIP (inner product) because embeddings are L2-normalised.
This makes inner product equivalent to cosine similarity.

Why FAISS?

• extremely fast vector search
• designed for dense semantic embeddings
• widely used in production retrieval systems

For this dataset (~20k documents), a flat index is appropriate because:
    - exact search improves result quality
    - dataset size is small enough that approximate indexing is unnecessary
"""

import os
import numpy as np
import faiss

from config import (
    FAISS_INDEX_PATH,
    DOCSTORE_PATH,
    LABELS_PATH,
    EMBEDDING_DIM,
    TOP_K,
)


class VectorStore:

    def __init__(self):

        self.index: faiss.IndexFlatIP | None = None
        self.documents: list[str] = []
        self.labels: list[str] = []

    # --------------------------------------------------------
    # BUILD INDEX
    # --------------------------------------------------------

    def build(self, embeddings: np.ndarray, documents: list[str], labels: list[str]):
        """
        Build FAISS index from pre-computed embeddings.

        embeddings must be L2-normalised so that inner-product equals cosine similarity.
        """

        if embeddings.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, "
                f"got {embeddings.shape[1]}"
            )

        if len(documents) != embeddings.shape[0]:
            raise ValueError("Documents and embeddings count mismatch")

        print("[VectorStore] Building FAISS index...")

        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)

        self.index.add(embeddings)

        self.documents = documents
        self.labels = labels

        print(f"[VectorStore] Indexed {self.index.ntotal} documents")

    # --------------------------------------------------------
    # SAVE INDEX
    # --------------------------------------------------------

    def save(self):

        if self.index is None:
            raise RuntimeError("Index has not been built")

        os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

        faiss.write_index(self.index, FAISS_INDEX_PATH)

        np.save(DOCSTORE_PATH, np.array(self.documents, dtype=object))
        np.save(LABELS_PATH, np.array(self.labels, dtype=object))

        print(f"[VectorStore] Saved index → {FAISS_INDEX_PATH}")

    # --------------------------------------------------------
    # LOAD INDEX
    # --------------------------------------------------------

    def load(self):

        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError("FAISS index not found")

        self.index = faiss.read_index(FAISS_INDEX_PATH)

        self.documents = np.load(DOCSTORE_PATH, allow_pickle=True).tolist()
        self.labels = np.load(LABELS_PATH, allow_pickle=True).tolist()

        print(f"[VectorStore] Loaded {self.index.ntotal} vectors")

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = TOP_K,
        filter_label: str | None = None,
    ):
        """
        Search the FAISS index.

        Parameters
        ----------
        query_vec : np.ndarray
            L2-normalised query embedding.

        top_k : int
            Number of results to return.

        filter_label : optional str
            Restrict retrieval to a specific newsgroup category.

        Returns
        -------
        list[dict]
        """

        if self.index is None:
            raise RuntimeError("Vector index not loaded")

        q = query_vec.reshape(1, -1)

        scores, indices = self.index.search(q, top_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            label = self.labels[idx]

            if filter_label is not None and label != filter_label:
                continue

            results.append(
                {
                    "text": self.documents[idx],
                    "label": label,
                    "score": float(score),
                }
            )

        return results