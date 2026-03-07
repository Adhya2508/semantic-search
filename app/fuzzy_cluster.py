"""
Runtime cluster inference — KMeans soft membership.

Prediction uses the same softmax-over-cosine-distance approach used
during training. No PCA needed since we work directly in embedding space.
"""

import numpy as np
from sklearn.preprocessing import normalize
from config import CLUSTER_MODEL_PATH


class FuzzyClusterer:

    def __init__(self):
        self.centroids   = None   # (n_clusters, 384)
        self.u           = None   # (n_clusters, n_docs)
        self.n_clusters  = None
        self.temperature = 0.3

    def load(self):
        data = np.load(CLUSTER_MODEL_PATH, allow_pickle=True).item()
        self.centroids   = data["centroids"].astype(np.float32)
        self.u           = data["u"]
        self.n_clusters  = data["n_clusters"]
        self.temperature = data.get("temperature", 0.3)
        print(f"[FuzzyClusterer] loaded {self.n_clusters} clusters "
              f"(method={data.get('method','kmeans_soft')})")

    def predict(self, vec: np.ndarray) -> np.ndarray:
        """
        Returns membership distribution (n_clusters,) for a query vector.
        Input vec must be L2-normalised (guaranteed by Embedder).
        """
        vec_norm = normalize(vec.reshape(1, -1), norm="l2")[0].astype(np.float32)

        # Cosine similarities to all centroids
        sims  = self.centroids @ vec_norm           # (n_clusters,)
        dists = 1.0 - sims

        # Softmax with temperature
        logits  = -dists / self.temperature
        logits -= logits.max()
        exp_l   = np.exp(logits)
        return exp_l / exp_l.sum()

    def dominant_cluster(self, vec: np.ndarray) -> int:
        return int(np.argmax(self.predict(vec)))