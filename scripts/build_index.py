"""
Build FAISS index from 20-Newsgroups corpus.

Cleaning strategy (each decision justified):

1. Strip full Usenet header block (everything before first blank line).
   Headers contain From/Organization/Path/NNTP routing metadata that
   causes the embedder to cluster by server or poster identity, not topic.

2. Remove "In article X writes:" attribution lines.
   These appear in EVERY post across ALL 20 categories, making them the
   dominant signal in raw embeddings. FCM/KMeans finds "reply style"
   clusters instead of topic clusters without this step.

3. Drop quoted lines ("> ").
   Quoted content duplicates another document's text and inflates
   cross-document similarity artificially.

4. Drop signature blocks ("--" separator).
   Signatures are personal metadata, not topic content.

5. Drop stray header fields that bleed past the blank-line split.
   Some posts have malformed headers that appear in the body.

6. Drop lines with >65% stop words.
   Lines like "I don't think people just like to know" are pure
   conversational filler. Without this filter, a garbage cluster forms
   (keywords: people, don, think, just, like) that absorbs ~10% of the
   corpus and obscures real topic structure.

7. Minimum 5 alphabetic words per line, minimum 50 words total per doc.
   Very short posts are almost always "me too" replies with no topical
   content. They add noise without signal.

8. MD5 deduplication.
   The dataset contains reposted articles. Duplicates inflate cluster
   sizes and skew membership distributions.

Usage:
    python -m scripts.build_index
"""

import os, re, sys, hashlib
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../..")))

import numpy as np
from app.embedder     import Embedder
from app.vector_store import VectorStore
from config           import DATASET_PATH, BATCH_SIZE, MODELS_DIR


# ── Compiled regex patterns (compiled once for performance) ───────────────────

# "In article <anything> writes:" — single biggest noise source
# Appears in every post across all categories
_RE_IN_ARTICLE = re.compile(r"in article\b.*?\bwrites\s*:", re.IGNORECASE)

# Standalone attribution: "John Smith wrote:" or "John Smith writes:"
_RE_ATTR_LINE  = re.compile(r"^[\w\s\.\,\-]{0,60}\bwrote?\s*:$", re.IGNORECASE)

# Header fields that sometimes bleed past the blank-line split
_RE_HEADER = re.compile(
    r"^(From|Subject|Organization|Lines|NNTP-Posting-Host|Article-I\.D\.|"
    r"Message-ID|References|Date|Xref|Path|Newsgroups|Reply-To|Sender|"
    r"Distribution|Keywords|Summary|Approved|Followup-To|Originator|"
    r"X-Mailer|X-Newsreader|Content-Type|Mime-Version|Posted)\s*:",
    re.IGNORECASE,
)

_RE_EMAIL     = re.compile(r"\S+@\S+\.\S+")
_RE_URL       = re.compile(r"https?://\S+|www\.\S+|ftp://\S+")
_RE_NO_WORDS  = re.compile(r"^[^a-zA-Z]*$")
_RE_SEPARATOR = re.compile(r"^[\-_=\*#~]{3,}$")
_RE_SPACES    = re.compile(r"\s+")

# Stop words for filler-line detection (Fix 4: eliminates garbage clusters)
# These are words that carry zero topical signal — pure conversational glue.
# A line that is >65% these words contributes noise, not topic signal.
_STOP = {
    "the","a","an","is","it","in","of","to","and","or","but","not","that",
    "this","with","for","on","are","be","as","at","by","from","have","has",
    "was","were","they","their","them","we","our","you","your","he","she",
    "his","her","its","just","don","doesnt","didnt","wont","can","could",
    "would","should","like","know","think","people","dont","i","my","me",
    "do","so","if","about","what","when","how","why","get","got","use",
    "used","using","also","more","some","than","there","been","will","one",
    "new","any","all","no","up","out","who","said","say","says","re","ve",
    "ll","im","ive","id","its","thats","theyre","youre","well","still",
    "even","only","very","really","much","many","same","other","another",
    "those","these","then","now","here","where","while","after","before",
    "because","though","although","however","therefore","thus","hence",
}


def clean(text: str) -> str:
    # ── 1. Strip header block ─────────────────────────────────────────────────
    parts = text.split("\n\n", 1)
    body  = parts[1] if len(parts) > 1 else text

    # ── 2. Remove "In article X writes:" inline occurrences ──────────────────
    body = _RE_IN_ARTICLE.sub(" ", body)

    # ── 3. Line-by-line filtering ─────────────────────────────────────────────
    lines = []
    for line in body.splitlines():
        s = line.strip()

        if not s:
            continue
        if s.startswith(">"):           # quoted reply
            continue
        if s.startswith("--"):          # signature separator
            continue
        if _RE_SEPARATOR.match(s):      # decorative separator line
            continue
        if _RE_HEADER.match(s):         # stray header field
            continue
        if _RE_ATTR_LINE.match(s):      # "Smith wrote:" attribution
            continue
        if _RE_NO_WORDS.match(s):       # no alphabetic characters
            continue

        # Remove emails and URLs within the line (keep surrounding text)
        s = _RE_EMAIL.sub(" ", s)
        s = _RE_URL.sub(" ", s)

        # Require at least 5 real words per line
        words = re.findall(r"[a-zA-Z]{3,}", s)
        if len(words) < 5:
            continue

        lines.append(s)

    # ── 4. Stop-word filter — removes filler lines ───────────────────────────
    # Lines where >65% of tokens are stop words carry no topical signal.
    # Without this, a garbage cluster forms with keywords like
    # "people don think just like" that absorbs ~10% of the corpus.
    filtered = []
    for line in lines:
        tokens     = re.findall(r"[a-zA-Z]+", line.lower())
        if not tokens:
            continue
        stop_ratio = sum(1 for w in tokens if w in _STOP) / len(tokens)
        if stop_ratio > 0.65:
            continue
        filtered.append(line)
    lines = filtered

    cleaned = _RE_SPACES.sub(" ", " ".join(lines)).strip()
    return cleaned


# ── Corpus loader ─────────────────────────────────────────────────────────────

def load_newsgroups(root: str):
    """
    Expects the classic 20-Newsgroups directory layout:
        root/
            alt.atheism/
            comp.graphics/
            ...  (20 category folders, each containing raw post files)
    """
    documents, labels = [], []
    seen = set()

    for category in sorted(os.listdir(root)):
        cat_path = os.path.join(root, category)
        if not os.path.isdir(cat_path):
            continue

        cat_count = 0
        for fname in os.listdir(cat_path):
            try:
                with open(os.path.join(cat_path, fname),
                          encoding="utf-8", errors="ignore") as f:
                    raw = f.read()
            except Exception:
                continue

            cleaned = clean(raw)

            # Minimum 50 words after cleaning.
            # Shorter posts are almost always "me too" replies or
            # administrative messages with no topical content.
            if len(cleaned.split()) < 50:
                continue

            # MD5 deduplication — dataset contains reposted articles
            h = hashlib.md5(cleaned.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)

            documents.append(cleaned)
            labels.append(category)
            cat_count += 1

        print(f"  {category:<35} {cat_count} docs")

    return documents, labels


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    print("Loading and cleaning corpus …")
    docs, labels = load_newsgroups(DATASET_PATH)
    print(f"\nTotal: {len(docs)} docs across {len(set(labels))} categories\n")

    # Sanity check — print 3 cleaned samples to verify noise is gone
    print("── Cleaned sample docs (verify no 'writes:', headers, quotes) ──")
    for i in [0, len(docs)//2, -1]:
        print(f"\n[{labels[i]}]\n{docs[i][:400]}\n")

    print("Embedding corpus …")
    embedder   = Embedder()
    # normalize_embeddings=True in Embedder means these are already L2-normed
    embeddings = embedder.encode(docs, batch_size=BATCH_SIZE)
    print(f"  Shape: {embeddings.shape}")

    print("Building FAISS index …")
    vs = VectorStore()
    vs.build(embeddings, docs, labels)
    vs.save()

    # Save embeddings separately for clustering script
    os.makedirs(MODELS_DIR, exist_ok=True)
    np.save(os.path.join(MODELS_DIR, "embeddings.npy"), embeddings)
    np.save(os.path.join(MODELS_DIR, "raw_labels.npy"),
            np.array(labels, dtype=object))

    print("\nCorpus preparation complete.")


if __name__ == "__main__":
    main()