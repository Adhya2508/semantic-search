# Semantic Search System — 20 Newsgroups

A lightweight semantic search pipeline over the 20 Newsgroups corpus featuring fuzzy clustering, a custom semantic cache, and a FastAPI service.

---

## Architecture Overview

```
Raw Corpus (20 Newsgroups)
        │
        ▼
┌─────────────────────┐
│   build_index.py    │  Cleaning → Embedding → FAISS Index
│                     │  all-MiniLM-L6-v2 (local, no download)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  build_clusters.py  │  KMeans + Soft Membership → Fuzzy Clusters
│                     │  Auto-selects k via silhouette sweep
└─────────────────────┘
        │
        ▼
┌──────────────────────┐
│ threshold_analysis.py│  Empirically derives cache threshold
│                      │  Auto-updates config.py
└──────────────────────┘
        │
        ▼
┌─────────────────────┐
│     FastAPI          │  /query  /cache/stats  DELETE /cache
│  + SemanticCache     │  Cluster-bucketed, thread-safe, LRU eviction
└─────────────────────┘
```

---

## Project Structure

```
semantic_search/
├── app/
│   ├── __init__.py
│   ├── api.py              # FastAPI endpoints
│   ├── embedder.py         # SentenceTransformer singleton
│   ├── fuzzy_cluster.py    # KMeans soft membership inference
│   ├── semantic_cache.py   # Custom semantic cache (no Redis)
│   └── vector_store.py     # FAISS index wrapper
├── scripts/
│   ├── build_index.py      # Step 1: embed + index corpus
│   ├── build_clusters.py   # Step 2: fit clusters + temperature analysis
│   └── threshold_analysis.py  # Step 3: derive + auto-write threshold
├── models/                 # Persisted artefacts (generated, not committed)
├── data/                   # 20 Newsgroups dataset (not committed)
├── config.py
├── main.py
└── requirements.txt
```

---

## Part 1 — Embedding & Vector Database

**Model:** `all-MiniLM-L6-v2` (local, 80MB, no GPU needed)

Chosen over larger models because:
- 384-dim embeddings: half the storage/compute vs 768-dim models
- Designed for semantic similarity, not generation
- Sufficient for retrieval tasks (MTEB rank ~40 vs ~35 for models 3× the cost)

**Cleaning decisions (each justified in code comments):**

| Decision | Reason |
|---|---|
| Strip Usenet headers | Prevent clustering by server/poster identity |
| Remove `In article X writes:` | Appears in every post — #1 noise source |
| Drop quoted lines `>` | Duplicates another document's content |
| Drop lines >65% stop words | Eliminates generic debate filler clusters |
| Minimum 50 words per doc | Short posts are almost always "me too" replies |
| MD5 deduplication | ~4% of corpus are reposted articles |

**Vector store:** FAISS `IndexFlatIP` (inner product = cosine for L2-normalised vectors). Exact search chosen over approximate because 14k docs fits comfortably — approximate indexing adds complexity with no benefit at this scale.

---

## Part 2 — Fuzzy Clustering

**Why not FCM?**

Fuzzy C-Means was tested first. It produced uniform membership scores (all ≈ 1/k) regardless of dimensionality reduction, cleaning quality, or parameter tuning. This is mathematically expected: sentence transformer embeddings lie on a uniform hypersphere (by design, for retrieval quality), and FCM's membership update rule degenerates to 1/k when all pairwise distances are equal — which they are in high-dimensional spherical space.

**Solution: KMeans + soft membership via softmax**

KMeans uses hard assignment steps (nearest centroid), making it robust to hyperspherical distribution. After fitting, we convert to fuzzy memberships using softmax over cosine distances:

```
membership_c = exp(-dist_c / T) / Σ exp(-dist_j / T)
```

**Temperature parameter T** controls peakedness:

| T | Behaviour |
|---|---|
| 0.05 | Near-hard assignment. Core docs: 0.95+ on dominant cluster |
| **0.10** | **Chosen.** Core docs: 0.70–0.85. Boundary docs: genuine spread across 2–3 clusters |
| 0.30 | Too soft. Max ~0.45. Core and boundary look the same |
| 1.00 | Approaches uniform 1/k. Clustering breaks down |

**Cluster count:** Auto-selected by silhouette sweep over k=10–25. Best k chosen by highest silhouette score, verified with Davies-Bouldin.

**Results (k=12, T=0.10):**

| Cluster | Keywords | Dominant membership |
|---|---|---|
| C3 | gun, fbi, government, batf | 0.37 |
| C4 | hockey, nhl, team, games | 0.46 |
| C5 | baseball, game, hit, players | 0.43 |
| C7 | key, encryption, clipper, nsa | 0.44 |
| C8 | doctor, medical, patients, disease | 0.37 |
| C9 | god, jesus, bible, christian | 0.37 |
| C10 | israel, jews, arab, armenian | 0.43 |

Boundary documents (highest entropy) correctly show split membership — e.g. a post about gun legislation scores C3:0.38 \| C11:0.29 \| C9:0.18, reflecting genuine multi-topic content.

---

## Part 3 — Semantic Cache

**Built entirely from scratch. No Redis, Memcached, or caching libraries.**

### Data structure

```python
_store:   list[CacheEntry]          # source of truth
_buckets: dict[int, list[int]]      # cluster_id → indices into _store
```

Cluster-bucketed lookup gives **O(bucket_size)** instead of O(N). With k=12 clusters and N=1000 entries, that is a ~12× speedup. As the cache grows large, this matters — a linear scan over thousands of embeddings would make the cache slower than recomputing.

### Similarity metric

Cosine similarity via dot product. Embeddings are L2-normalised (unit vectors), so `cosine(a,b) = dot(a,b)`. Avoids recomputing norms on every comparison.

### Threshold — the key tunable parameter

`CACHE_SIMILARITY_THRESHOLD` defines semantic equivalence:

```
sim(query, cached_query) >= threshold  →  cache HIT  (reuse result)
sim(query, cached_query) <  threshold  →  cache MISS (recompute)
```

**Empirically derived by `scripts/threshold_analysis.py`** (auto-writes to config.py):

```
Paraphrase pairs:       0.603 – 0.707   (semantically equivalent)
Different-topic pairs:  0.005 – 0.061   (completely unrelated)
Separation gap:         0.541           (clean, no ambiguous zone)
Chosen threshold:       0.58
```

What each zone reveals:

- **Below 0.06** — Collision zone. Unrelated queries return each other's results.
- **0.06–0.60** — Dead zone. No real queries land here. Clean model separation.
- **0.60–0.71** — Paraphrase zone. All equivalent queries land here.
- **Above 0.71** — Nothing hits. Cache degenerates to pure vector search.

The threshold is model-dependent, not a magic number. `0.58` for MiniLM = `0.82` for MPNet — identical behaviour, different scales. Re-run `threshold_analysis.py` after any model change and config.py updates automatically.

### Additional features

- **LRU eviction** — oldest entry removed when cache exceeds `max_size=10,000`
- **Thread safety** — `threading.Lock` around lookup→miss→insert sequence prevents duplicate insertions from concurrent requests
- **Runtime override** — `CACHE_SIMILARITY_THRESHOLD=0.65 uvicorn ...` for A/B testing without redeploying

---

## Part 4 — FastAPI Service

### Endpoints

**POST /query**
```json
{
  "query": "Which cryptographic algorithms does the NSA employ?",
  "cache_hit": true,
  "matched_query": "What encryption does the NSA use?",
  "similarity_score": 0.8052,
  "dominant_cluster": 3,
  "result": { "top_results": [...] }
}
```

**GET /cache/stats**
```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

**DELETE /cache** — Flushes cache and resets all stats.

**GET /health** — Simple health check.

---

## Setup & Running

### Requirements

- Python 3.10+
- `all-MiniLM-L6-v2` model downloaded locally
- 20 Newsgroups dataset extracted to `data/20_newsgroups/`

### Installation

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### One-time pipeline (run in order)

```bash
# Step 1 — Clean, embed, index (~20-25 min on CPU)
python -m scripts.build_index

# Step 2 — Cluster + temperature analysis (~15 min)
python -m scripts.build_clusters

# Step 3 — Derive threshold + auto-update config (~30 sec)
python -m scripts.threshold_analysis
```

### Start the API

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

### Test it

```bash
# First query — cache MISS
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What encryption does the NSA use?"}'

# Paraphrase — cache HIT (similarity ~0.80)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Which cryptographic algorithms does the NSA employ?"}'

# Stats
curl http://localhost:8000/cache/stats

# Clear cache
curl -X DELETE http://localhost:8000/cache
```

---

## Docker (Optional)

```bash
docker build -t semantic-search .
docker run -v E:/hf_models:/hf_models -p 8000:8000 semantic-search
```

---

## Key Design Decisions Summary

| Decision | Choice | Reason |
|---|---|---|
| Embedding model | all-MiniLM-L6-v2 | Fast on CPU, sufficient for retrieval |
| Vector store | FAISS IndexFlatIP | Exact search justified at 14k docs |
| Clustering algorithm | KMeans + softmax | FCM collapses on hyperspherical embeddings |
| Fuzziness mechanism | Softmax temperature | Interpretable, tunable, avoids FCM collapse |
| Cache lookup | Cluster-bucketed | O(N/k) vs O(N) — scales with cache size |
| Threshold selection | Empirical gap analysis | Model-dependent, not hardcoded |
| Thread safety | threading.Lock | FastAPI runs sync endpoints in threads |

---

## Dataset

20 Newsgroups — ~20,000 Usenet posts across 20 topic categories.  
Source: https://archive.ics.uci.edu/dataset/113/twenty+newsgroups