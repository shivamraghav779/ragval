"""Retrieval diversity: how non-redundant are the retrieved chunks?

Pure text, no LLM. A retriever that returns five near-duplicate chunks wastes
context-window budget and can bias the generator. Diversity = 1 - mean pairwise
TF-IDF cosine similarity between chunks.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.utils import text


class RetrievalDiversityMetric(BaseMetric):
    name = "retrieval_diversity"
    description = "Non-redundancy of retrieved chunks (pure text, no LLM)"
    requires_ground_truth = False
    category = "retrieval"

    async def _compute(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        provider: Any,
        ground_truth: Optional[str] = None,
        **kwargs: Any,
    ) -> MetricResult:
        if len(contexts) < 2:
            return self._na("need at least 2 chunks to measure diversity")

        vectors = text.compute_tfidf(contexts)
        sims = [
            text.cosine_similarity(vectors[i], vectors[j])
            for i, j in combinations(range(len(contexts)), 2)
        ]
        mean_sim = sum(sims) / len(sims)
        score = max(0.0, 1.0 - mean_sim)

        # Flag the most redundant pairs.
        pairs = sorted(
            (
                {"a": i, "b": j, "similarity": text.cosine_similarity(vectors[i], vectors[j])}
                for i, j in combinations(range(len(contexts)), 2)
            ),
            key=lambda p: p["similarity"],
            reverse=True,
        )
        redundant = [p for p in pairs if p["similarity"] > 0.8]

        return MetricResult(
            score=score,
            reasoning=(
                f"Mean pairwise chunk similarity {mean_sim:.3f} across "
                f"{len(contexts)} chunks. Diversity = {score:.3f}."
            ),
            violations=[
                f"chunk[{p['a']}] and chunk[{p['b']}] are near-duplicates "
                f"(sim {p['similarity']:.2f})"
                for p in redundant
            ],
            metadata={
                "mean_pairwise_similarity": mean_sim,
                "redundant_pairs": redundant,
                "chunk_count": len(contexts),
            },
            metric_name=self.name,
        )
