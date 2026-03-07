"""
FastAPI service exposing the semantic search system.

Heavy components are initialised once at server startup:

    • Embedder        → SentenceTransformer model
    • VectorStore     → FAISS index
    • FuzzyClusterer  → cluster model
    • SemanticCache   → query cache

This avoids expensive reloads for every request and keeps latency low.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from app.embedder import Embedder
from app.vector_store import VectorStore
from app.fuzzy_cluster import FuzzyClusterer
from app.semantic_cache import SemanticCache


# ---------------------------------------------------------
# STARTUP / SHUTDOWN
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialise heavy resources once when the API starts.
    """

    print("[API] Initialising components...")

    app.state.embedder = Embedder()

    app.state.vs = VectorStore()
    app.state.vs.load()

    app.state.clusterer = FuzzyClusterer()
    app.state.clusterer.load()

    app.state.cache = SemanticCache()

    print("[API] System ready.")

    yield

    print("[API] Shutting down.")


app = FastAPI(
    title="Semantic Search API",
    description="Cluster-aware semantic search with fuzzy clustering and semantic caching",
    lifespan=lifespan
)


# ---------------------------------------------------------
# REQUEST SCHEMA
# ---------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    filter_label: str | None = None


# ---------------------------------------------------------
# QUERY ENDPOINT
# ---------------------------------------------------------

@app.post("/query")
def query_endpoint(req: QueryRequest):

    embedder = app.state.embedder
    vs = app.state.vs
    clusterer = app.state.clusterer
    cache = app.state.cache

    query = req.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # -----------------------------------------------------
    # STEP 1 — EMBED QUERY
    # -----------------------------------------------------

    q_vec = embedder.encode(query)

    # -----------------------------------------------------
    # STEP 2 — PREDICT DOMINANT CLUSTER
    # -----------------------------------------------------

    dom_cluster = clusterer.dominant_cluster(q_vec)

    # -----------------------------------------------------
    # STEP 3 — CHECK SEMANTIC CACHE
    # -----------------------------------------------------

    cached_entry, sim_score = cache.lookup(q_vec, dom_cluster)

    if cached_entry is not None:

        return {
            "query": query,
            "cache_hit": True,
            "matched_query": cached_entry.query,
            "similarity_score": round(sim_score, 4),
            "dominant_cluster": dom_cluster,
            "result": cached_entry.result,
        }

    # -----------------------------------------------------
    # STEP 4 — VECTOR SEARCH (CACHE MISS)
    # -----------------------------------------------------

    results = vs.search(
        q_vec,
        filter_label=req.filter_label
    )

    result_payload = {
        "top_results": [
            {
                "text": r["text"][:300],
                "label": r["label"],
                "score": round(r["score"], 4)
            }
            for r in results
        ]
    }

    # -----------------------------------------------------
    # STEP 5 — STORE IN CACHE
    # -----------------------------------------------------

    cache.store(query, q_vec, dom_cluster, result_payload)

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": round(sim_score, 4),
        "dominant_cluster": dom_cluster,
        "result": result_payload,
    }


# ---------------------------------------------------------
# CACHE STATS
# ---------------------------------------------------------

@app.get("/cache/stats")
def cache_stats():
    """
    Returns cache usage statistics.
    """
    return app.state.cache.stats


# ---------------------------------------------------------
# CLEAR CACHE
# ---------------------------------------------------------

@app.delete("/cache")
def clear_cache():
    """
    Reset cache contents.
    """
    app.state.cache.flush()

    return {"message": "Cache cleared successfully."}


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------

@app.get("/health")
def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "ok"}