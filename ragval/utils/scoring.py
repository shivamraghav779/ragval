"""Numeric scoring helpers: means, clamping, and ranking metrics (DCG/NDCG/MRR)."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence


def weighted_mean(
    scores: Sequence[Optional[float]],
    weights: Optional[Sequence[float]] = None,
) -> float:
    """Weighted mean that ignores ``None`` entries.

    If ``weights`` is ``None`` a simple mean is used. If every score is ``None``
    (or the list is empty) the result is ``0.0``.
    """
    if not scores:
        return 0.0

    if weights is None:
        present = [s for s in scores if s is not None]
        if not present:
            return 0.0
        return sum(present) / len(present)

    if len(weights) != len(scores):
        raise ValueError("scores and weights must be the same length")

    num = 0.0
    den = 0.0
    for score, weight in zip(scores, weights):
        if score is None:
            continue
        num += score * weight
        den += weight
    if den == 0.0:
        return 0.0
    return num / den


def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp ``value`` into ``[min_val, max_val]``."""
    return max(min_val, min(max_val, value))


def safe_divide(num: float, den: float, default: float = 0.0) -> float:
    """Divide, returning ``default`` when the denominator is zero."""
    if den == 0:
        return default
    return num / den


def compute_dcg(relevances: List[float]) -> float:
    """Discounted cumulative gain: sum of ``rel_i / log2(i + 2)``."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def compute_ndcg(
    retrieved_relevances: List[float],
    ideal_relevances: List[float],
) -> float:
    """NDCG = DCG(retrieved) / DCG(sorted ideal). Returns 0.0 if ideal DCG is 0."""
    idcg = compute_dcg(sorted(ideal_relevances, reverse=True))
    if idcg == 0.0:
        return 0.0
    return compute_dcg(retrieved_relevances) / idcg


def compute_mrr(retrieved_relevances: List[bool]) -> float:
    """Reciprocal rank of the first relevant item (1-indexed). 0.0 if none."""
    for i, relevant in enumerate(retrieved_relevances):
        if relevant:
            return 1.0 / (i + 1)
    return 0.0


def compute_hit_rate(retrieved_relevances: List[bool], k: int) -> float:
    """1.0 if any of the first ``k`` items is relevant, else 0.0."""
    if k <= 0:
        return 0.0
    return 1.0 if any(retrieved_relevances[:k]) else 0.0
