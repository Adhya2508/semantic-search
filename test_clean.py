# test_clean.py  — run with: python test_clean.py
import os, sys
sys.path.insert(0, ".")
from scripts.build_index import clean

# Pick any file from your dataset
test_file = r"data/20_newsgroups/talk.politics.guns/54219"  # adjust path

with open(test_file, encoding="utf-8", errors="ignore") as f:
    raw = f.read()

print("=== RAW (first 500 chars) ===")
print(raw[:500])
print("\n=== CLEANED ===")
print(clean(raw))