import os

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_PATH    = r"E:\hf_models\all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ── Dataset ───────────────────────────────────────────────────────────────────
# Root folder containing one subfolder per 20NG category.
# Layout: data/20_newsgroups/alt.atheism/, data/20_newsgroups/comp.graphics/, ...
DATASET_PATH  = r"data/20_newsgroups"

# ── Persisted artefacts ───────────────────────────────────────────────────────
MODELS_DIR         = "models"
FAISS_INDEX_PATH   = os.path.join(MODELS_DIR, "faiss_index.bin")
DOCSTORE_PATH      = os.path.join(MODELS_DIR, "docstore.npy")
LABELS_PATH        = os.path.join(MODELS_DIR, "labels.npy")
CLUSTER_MODEL_PATH = os.path.join(MODELS_DIR, "cluster_model.npy")

# ── Embedding / FAISS ─────────────────────────────────────────────────────────
BATCH_SIZE = 128
TOP_K      = 5

# ── Fuzzy clustering ──────────────────────────────────────────────────────────
# N_CLUSTERS is auto-selected by build_clusters.py silhouette sweep.
# This fallback is only used if FuzzyClusterer.fit() is called directly.
N_CLUSTERS  = 20
FCM_ERROR   = 0.005
FCM_MAXITER = 300
PCA_COMPONENTS = 100

# ── Semantic cache ────────────────────────────────────────────────────────────
# AUTO-MANAGED — do not edit this line manually.
# Value is computed and written by scripts/threshold_analysis.py based on
# the empirical similarity geometry of the configured embedding model.
# To recalibrate: python -m scripts.threshold_analysis
#
# How it works:
#   threshold = paraphrase_floor - safety_margin
#   where paraphrase_floor = lowest cosine similarity between known paraphrase pairs
#   and safety_margin      = max(0.02, 5% of the paraphrase/diff-topic gap)
#
# For all-MiniLM-L6-v2:
#   paraphrase range  : 0.60 – 0.71
#   diff-topic range  : 0.00 – 0.06
#   separation gap    : 0.54  (clean, no ambiguous zone)
#   computed threshold: 0.58
#
# Override at runtime via environment variable (useful for A/B testing):
#   CACHE_SIMILARITY_THRESHOLD = 0.58 uvicorn app.api:app --port 8000
CACHE_SIMILARITY_THRESHOLD = float(
    os.environ.get("CACHE_SIMILARITY_THRESHOLD", "0.58")
)