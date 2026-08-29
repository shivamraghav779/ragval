"""Pure-Python text processing. Standard library only.

No numpy, no sklearn, no sentence-transformers. Everything here is deliberately
lightweight so that ``pip install ragval`` stays fast and dependency-light.
TF-IDF cosine similarity stands in for embedding similarity; it is weaker but
requires no model download and no API call.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Set

# Common English stopwords, per the specification.
STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought", "used",
}

# Abbreviations whose trailing period must not be treated as a sentence break.
_ABBREVIATIONS = {"dr", "vs", "etc", "al", "mr", "mrs", "ms", "prof", "inc", "fig", "e.g", "i.e"}

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")

_UNIT_RE = re.compile(
    r"\b\d+\.?\d*\s*(mg|mcg|g|kg|ml|l|mmol|meq|iu|u|dl|mmhg|bpm)\b",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(r"\d+\.?\d*\s*%")
_CURRENCY_RE = re.compile(r"[$£€₹]\s?\d[\d,]*\.?\d*")
_DATE_RE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}/\d{1,2}/\d{2,4}"
    r"|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b"
)
_DRUG_RE = re.compile(r"\b([A-Z][a-z]{3,})\b(?:\s+\d+\.?\d*\s*(?:mg|mcg|g|ml|units|IU))?")
_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")


def tokenize(text: str) -> List[str]:
    """Lowercase, drop punctuation (keeping hyphens in compounds), remove stopwords."""
    if not text:
        return []
    tokens = [m.group(0).lower() for m in _WORD_RE.finditer(text)]
    return [t for t in tokens if t not in STOPWORDS]


def compute_tfidf(corpus: List[str]) -> List[Dict[str, float]]:
    """Return one ``{term: tfidf}`` dict per document in ``corpus``."""
    if not corpus:
        return []
    tokenized = [tokenize(doc) for doc in corpus]
    total_docs = len(tokenized)

    doc_freq: Counter = Counter()
    for tokens in tokenized:
        for term in set(tokens):
            doc_freq[term] += 1

    idf: Dict[str, float] = {
        term: math.log(total_docs / (1 + df)) for term, df in doc_freq.items()
    }

    vectors: List[Dict[str, float]] = []
    for tokens in tokenized:
        length = len(tokens)
        if length == 0:
            vectors.append({})
            continue
        counts = Counter(tokens)
        vectors.append(
            {term: (count / length) * idf.get(term, 0.0) for term, count in counts.items()}
        )
    return vectors


def cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    """Sparse cosine similarity. Returns 0.0 if either vector is empty."""
    if not vec_a or not vec_b:
        return 0.0
    common = set(vec_a) & set(vec_b)
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def sentence_similarity(text_a: str, text_b: str) -> float:
    """TF-IDF cosine similarity between two texts, using them as a two-doc corpus."""
    if not text_a or not text_b:
        return 0.0
    vectors = compute_tfidf([text_a, text_b])
    if len(vectors) != 2:
        return 0.0
    return cosine_similarity(vectors[0], vectors[1])


def extract_entities(text: str) -> List[Dict[str, object]]:
    """Regex-based entity extraction. Returns ``[{text, type, position}]``."""
    if not text:
        return []
    found: List[Dict[str, object]] = []

    def add(match: "re.Match[str]", etype: str) -> None:
        found.append(
            {"text": match.group(0).strip(), "type": etype, "position": match.start()}
        )

    for m in _UNIT_RE.finditer(text):
        add(m, "numbers_with_units")
    for m in _PERCENT_RE.finditer(text):
        add(m, "percentage")
    for m in _CURRENCY_RE.finditer(text):
        add(m, "currency")
    for m in _DATE_RE.finditer(text):
        add(m, "date")
    for m in _PROPER_NOUN_RE.finditer(text):
        add(m, "proper_noun")
    for m in _DRUG_RE.finditer(text):
        # Skip words that begin a sentence and are common words; keep the rest.
        found.append(
            {"text": m.group(1).strip(), "type": "drug_candidate", "position": m.start()}
        )

    # De-duplicate on (lowercased text, type) while preserving first position.
    seen: Set[tuple] = set()
    unique: List[Dict[str, object]] = []
    for e in sorted(found, key=lambda x: x["position"]):
        key = (str(e["text"]).lower(), e["type"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(e)
    return unique


def extract_sentences(text: str) -> List[str]:
    """Split into sentences on ``.!?`` boundaries, preserving known abbreviations."""
    if not text:
        return []
    # Protect abbreviations by temporarily replacing their period.
    protected = text
    for abbr in _ABBREVIATIONS:
        protected = re.sub(
            rf"\b({re.escape(abbr)})\.",
            r"\1<DOT>",
            protected,
            flags=re.IGNORECASE,
        )
    pieces = re.split(r"(?<=[.!?])\s+", protected)
    result = []
    for piece in pieces:
        cleaned = piece.replace("<DOT>", ".").strip()
        if cleaned:
            result.append(cleaned)
    return result


def compute_f1_overlap(set_a: Set[str], set_b: Set[str]) -> Dict[str, float]:
    """Token-level precision / recall / F1 between two sets."""
    set_a = set(set_a)
    set_b = set(set_b)
    if not set_a and not set_b:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not set_a or not set_b:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = len(set_a & set_b)
    precision = tp / len(set_a)
    recall = tp / len(set_b)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_ngrams(text: str, n: int) -> List[str]:
    """N-grams from tokenized text, joined by spaces."""
    tokens = tokenize(text)
    if n <= 0 or len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def deduplicate_preserving_order(items: Iterable) -> List:
    """Remove duplicates while preserving insertion order."""
    seen: Set = set()
    result: List = []
    for item in items:
        try:
            marker = item
            hash(marker)
        except TypeError:
            marker = repr(item)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(item)
    return result
