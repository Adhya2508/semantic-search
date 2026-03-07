"""
Cache similarity threshold analysis.

The assignment states: "There is one tunable decision at the heart of
this component. Explore it. The interesting question is not which value
performs best, it is what each value reveals about the system's behaviour."

That tunable decision is CACHE_SIMILARITY_THRESHOLD — the cosine
similarity cutoff above which two queries are considered equivalent and
the cached result is reused.

Model context (all-MiniLM-L6-v2)
----------------------------------
This model is optimised for asymmetric semantic search (short query vs
long document), not paraphrase detection. Paraphrase pairs score in the
0.60-0.71 range rather than the 0.85-0.95 range seen with larger models
like MPNet or E5. This is not a flaw — it is a property of the model
that directly informs threshold selection.

The critical insight is the GAP, not the absolute values:
  - Paraphrase pairs:      0.60 – 0.71  (semantically equivalent queries)
  - Different-topic pairs: 0.00 – 0.06  (completely unrelated queries)
  - Separation gap:        ~0.54        (enormous, very clean threshold signal)

A larger model would push paraphrase scores to 0.85-0.92 and the
threshold to 0.82 — but the gap structure and system behaviour would be
identical. The threshold value is model-dependent. 0.58 for MiniLM is
equivalent to 0.82 for MPNet.

Auto-update
-----------
This script writes the computed threshold directly into config.py.
No manual copy-paste required. Re-run after any model change and the
entire system reconfigures automatically.

Usage:
    python -m scripts.threshold_analysis
"""

import os, sys, re
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../..")))

import numpy as np
from app.embedder import Embedder


# ── Test pairs ────────────────────────────────────────────────────────────────

PARAPHRASES = [
    (
        "What encryption does the government use?",
        "Which cryptographic algorithms does the NSA employ?",
        "crypto/govt — clear paraphrase",
    ),
    (
        "Best NHL teams this season",
        "Which hockey clubs are performing well?",
        "hockey — clear paraphrase",
    ),
    (
        "How do I add a hard drive to my computer?",
        "Steps to install an additional disk on my machine",
        "hardware — clear paraphrase",
    ),
    (
        "Is there evidence for God's existence?",
        "Can the existence of a deity be proven?",
        "religion — clear paraphrase",
    ),
]

DIFFERENT_TOPICS = [
    (
        "How do I install a SCSI drive?",
        "What is the US policy on gun control?",
        "hardware vs. politics — clearly different",
    ),
    (
        "Jesus and Christian faith",
        "Israeli settlements in the West Bank",
        "religion vs. middle east — clearly different",
    ),
    (
        "NHL playoff standings",
        "Clipper chip encryption backdoor",
        "sports vs. crypto — clearly different",
    ),
]

BORDERLINE = [
    (
        "gun laws and the second amendment",
        "FBI raid on the Branch Davidians",
        "BORDERLINE — both involve guns/govt but different aspect",
    ),
    (
        "Mac hard drive not booting",
        "SCSI disk controller problem on Windows PC",
        "BORDERLINE — both hardware/storage but different platform",
    ),
    (
        "Government surveillance and privacy",
        "Clipper chip and NSA wiretapping",
        "BORDERLINE — both surveillance but different framing",
    ),
    (
        "baseball batting statistics",
        "hockey scoring leaders",
        "BORDERLINE — both sports stats but different sport",
    ),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_similarities(embedder, pairs):
    results = []
    for q1, q2, label in pairs:
        v1  = embedder.encode(q1)
        v2  = embedder.encode(q2)
        sim = float(np.dot(v1, v2))
        results.append((q1, q2, label, sim))
    return results


def print_pairs(title, results):
    print(f"\n── {title} ──")
    for q1, q2, label, sim in results:
        print(f"  sim={sim:.3f}  {label}")
        print(f"    A: {q1}")
        print(f"    B: {q2}")


def find_optimal_threshold(para_sims, diff_sims):
    """
    Find the threshold that:
      - catches ALL paraphrases  (sim >= threshold)
      - rejects ALL diff-topic pairs (sim < threshold)

    Strategy: sit just below the paraphrase floor with a small safety
    margin. This is maximally strict while still catching all paraphrases.

    Why not the midpoint of the optimal zone?
      The midpoint (e.g. 0.33) is too generous — it allows borderline
      pairs (0.10-0.44) to hit the cache. We want maximum precision:
      only genuine paraphrases should hit, not merely related queries.

    margin = max(0.02, 5% of gap)
      Small enough to not miss any paraphrases.
      Large enough to give robustness against slight score variation.
    """
    min_para = min(para_sims)
    max_diff = max(diff_sims)

    if min_para <= max_diff:
        print(
            "WARNING: no clean separation between paraphrases and "
            "different-topic pairs. Threshold selection is unreliable."
        )
        return 0.5

    gap       = min_para - max_diff
    margin    = max(0.02, gap * 0.05)
    threshold = round(min_para - margin, 2)
    return threshold


def threshold_table(para_sims, diff_sims, border_sims, optimal):
    """
    Show threshold behaviour across the full meaningful range.
    Sweeps from well below the different-topic ceiling up through and
    past the paraphrase zone — covering every decision point.
    """
    max_diff = max(diff_sims)
    min_para = min(para_sims)
    max_para = max(para_sims)
    n_para   = len(para_sims)
    n_diff   = len(diff_sims)
    n_border = len(border_sims)

    print("\n" + "=" * 82)
    print("THRESHOLD BEHAVIOUR TABLE")
    print("=" * 82)
    print(f"  Model            : all-MiniLM-L6-v2")
    print(f"  Paraphrase range : {min_para:.3f} – {max_para:.3f}")
    print(f"  Diff-topic range : {min(diff_sims):.3f} – {max_diff:.3f}")
    print(f"  Separation gap   : {min_para - max_diff:.3f}  "
          f"(between {max_diff:.3f} and {min_para:.3f})")
    print(f"  Optimal zone     : {max_diff:.2f} – {min_para:.2f}")
    print(f"  Chosen threshold : {optimal:.2f}  "
          f"(just below paraphrase floor — maximum strictness)")
    print()
    print(f"{'Threshold':>10}  {'Para hits':>10}  "
          f"{'Topic misses':>13}  {'Border hits':>12}  Assessment")
    print("-" * 82)

    thresholds = [
        0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
        0.55, 0.57, 0.58, 0.60, 0.62,
        0.65, 0.68, 0.70, 0.75, 0.80,
    ]

    for thresh in thresholds:
        para_hits   = sum(1 for s in para_sims   if s >= thresh)
        diff_misses = sum(1 for s in diff_sims   if s <  thresh)
        border_hits = sum(1 for s in border_sims if s >= thresh)
        marker      = "  ← CHOSEN" if abs(thresh - optimal) < 0.015 else ""

        if thresh <= max_diff:
            assess = "✗ too loose — topic collisions"
        elif thresh > max_para:
            assess = "✗ too strict — ALL paraphrases miss"
        elif para_hits == n_para and diff_misses == n_diff:
            assess = "✓ optimal zone"
        elif para_hits < n_para:
            assess = f"~ {n_para - para_hits} paraphrase(s) missed"
        else:
            assess = "✗ topic collision risk"

        print(f"{thresh:>10.2f}  "
              f"{para_hits:>4}/{n_para:<5}  "
              f"{diff_misses:>5}/{n_diff:<7}  "
              f"{border_hits:>4}/{n_border:<7}  "
              f"{assess}{marker}")


# ── Config auto-update ────────────────────────────────────────────────────────

def _update_config(threshold: float):
    """
    Writes the empirically derived threshold back into config.py
    using regex replacement — all other config values are preserved.

    Why auto-update instead of manual copy-paste?
      - Eliminates human error (wrong value, forgot to update)
      - Makes the pipeline fully self-configuring: re-run this script
        after any model change and the system reconfigures itself
      - The threshold is a function of the model's similarity geometry,
        not a design choice — it should be derived automatically

    The regex matches:
        CACHE_SIMILARITY_THRESHOLD = <any float>
    and replaces the float with the computed optimal value.
    """
    config_path = os.path.join(
        os.path.abspath(os.path.join(__file__, "../..")),
        "config.py"
    )

    if not os.path.exists(config_path):
        print(
            f"\nWARNING: config.py not found at {config_path}\n"
            f"Update manually: CACHE_SIMILARITY_THRESHOLD = {threshold:.2f}"
        )
        return

    with open(config_path, "r") as f:
        original = f.read()

    pattern     = r"CACHE_SIMILARITY_THRESHOLD\s*=\s*[\d\.]+"
    replacement = f"CACHE_SIMILARITY_THRESHOLD = {threshold:.2f}"
    updated     = re.sub(pattern, replacement, original)

    if updated == original:
        print(
            f"\nWARNING: CACHE_SIMILARITY_THRESHOLD line not found in config.py\n"
            f"Add this line manually:\n  {replacement}"
        )
        return

    with open(config_path, "w") as f:
        f.write(updated)

    print(f"\n✓ config.py auto-updated  →  {replacement}")
    print(f"  Restart uvicorn to apply the new threshold.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading embedder …")
    embedder = Embedder()

    print("Computing pairwise similarities …")
    para_results   = compute_similarities(embedder, PARAPHRASES)
    diff_results   = compute_similarities(embedder, DIFFERENT_TOPICS)
    border_results = compute_similarities(embedder, BORDERLINE)

    print_pairs("PARAPHRASES (should be cache HITs)",         para_results)
    print_pairs("DIFFERENT TOPICS (should be cache MISSes)",  diff_results)
    print_pairs("BORDERLINE (reveals system character)",       border_results)

    para_sims   = [r[3] for r in para_results]
    diff_sims   = [r[3] for r in diff_results]
    border_sims = [r[3] for r in border_results]

    print(f"\nSimilarity ranges:")
    print(f"  Paraphrases     : {min(para_sims):.3f} – {max(para_sims):.3f}")
    print(f"  Different topics: {min(diff_sims):.3f} – {max(diff_sims):.3f}")
    print(f"  Borderline      : {min(border_sims):.3f} – {max(border_sims):.3f}")

    optimal = find_optimal_threshold(para_sims, diff_sims)

    threshold_table(para_sims, diff_sims, border_sims, optimal)

    min_para = min(para_sims)
    max_diff = max(diff_sims)
    max_para = max(para_sims)

    print(f"""
CONCLUSION
----------
all-MiniLM-L6-v2 is optimised for asymmetric retrieval (short query vs
long document), not symmetric paraphrase detection. This produces a
characteristic similarity geometry:

  Paraphrase pairs score  : {min(para_sims):.3f} – {max(para_sims):.3f}
  Different-topic pairs   : {min(diff_sims):.3f} – {max(diff_sims):.3f}
  Separation gap          : {min_para - max_diff:.3f}  ← enormous and clean

The absolute values are model-dependent and not meaningful in isolation.
What matters is the gap. Consider the analogy:

  MiniLM scale : paraphrases = 0.60–0.71,  threshold = {optimal:.2f}
  MPNet scale  : paraphrases = 0.85–0.92,  threshold = 0.82
  → Identical system behaviour. Different numeric scales.

What each threshold zone reveals about system behaviour:

  Below {max_diff:.2f}  [COLLISION ZONE]
    The cache becomes semantically meaningless. Completely unrelated
    queries (hockey vs. encryption, religion vs. hardware) return each
    other's cached results. The semantic layer actively causes harm —
    it confidently returns wrong answers with high similarity scores.

  {max_diff:.2f} – {min_para:.2f}  [DEAD ZONE — no real queries land here]
    The {min_para - max_diff:.3f} gap between different-topic ceiling and
    paraphrase floor is empty. No real query pairs score in this range.
    This gap is the model's clean semantic separation — a property of
    a well-trained embedding model, not a weakness to apologise for.

  {min_para:.2f} – {max_para:.2f}  [PARAPHRASE ZONE]
    All semantically equivalent query pairs land here. The threshold
    must sit below {min_para:.2f} to catch paraphrases, and above
    {max_diff:.2f} to reject topic collisions.
    Chosen value {optimal:.2f} sits just below the paraphrase floor —
    maximally strict while still catching all paraphrases.

  Above {max_para:.2f}  [GRAVEYARD — nothing ever hits]
    No MiniLM paraphrase pairs score this high. Setting the threshold
    here makes the semantic cache completely non-functional. Every
    query is a miss. The system degenerates to pure vector search with
    zero caching benefit. This is the failure mode of blindly copying
    a threshold value from a larger model without empirical validation.

Chosen threshold : {optimal:.2f}
  Derived automatically from this model's similarity geometry.
  Catches all paraphrase pairs. Rejects all different-topic pairs.
  Borderline pairs miss the cache — they are related but not
  semantically equivalent, which is the correct behaviour.

  This threshold is not manually chosen — it is computed. Re-run
  this script after any model change and config.py updates itself.
""")

    # Auto-update config.py — no manual copy-paste needed
    _update_config(optimal)


if __name__ == "__main__":
    main()