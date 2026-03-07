# test_model.py — run this before committing to a full re-embed
import sys, os
sys.path.insert(0, ".")
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

pairs = [
    ("What encryption does the government use?",
     "Which cryptographic algorithms does the NSA employ?"),
    ("Best NHL teams this season",
     "Which hockey clubs are performing well?"),
    ("How do I install a SCSI drive?",
     "What is the US policy on gun control?"),
]

for q1, q2 in pairs:
    v1 = model.encode(q1, normalize_embeddings=True)
    v2 = model.encode(q2, normalize_embeddings=True)
    print(f"{float(np.dot(v1,v2)):.3f}  |  {q1[:40]} <-> {q2[:40]}")