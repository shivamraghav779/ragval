"""Normalized Discounted Cumulative Gain over graded chunk relevance."""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.utils import prompts
from ragval.utils.scoring import compute_dcg, compute_ndcg


async def judge_graded_relevance(
    question: str, contexts: List[str], provider: Any
) -> List[float]:
    """Ask the LLM for a 0-3 relevance grade per chunk, in index order."""
    data = await provider.complete_json(
        prompts.CHUNK_GRADED_RELEVANCE_PROMPT.format(
            question=question, contexts=prompts.join_contexts(contexts)
        )
    )
    grades = [0.0] * len(contexts)
    for entry in data.get("grades", []) or []:
        try:
            idx = int(entry.get("index"))
            grade = float(entry.get("grade", 0))
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(contexts):
            grades[idx] = max(0.0, min(3.0, grade))
    return grades


class NDCGMetric(BaseMetric):
    name = "ndcg"
    description = "Position-aware relevance. Rewards relevant chunks at top."
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
        if not contexts:
            return self._na("no contexts to rank")

        grades = kwargs.get("graded_relevances")
        if grades is None or len(grades) != len(contexts):
            grades = await judge_graded_relevance(question, contexts, provider)
        grades = [max(0.0, min(3.0, float(g))) for g in grades]

        dcg = compute_dcg(grades)
        idcg = compute_dcg(sorted(grades, reverse=True))
        ndcg = compute_ndcg(grades, grades)

        return MetricResult(
            score=ndcg,
            reasoning=(
                f"NDCG = {ndcg:.3f} (DCG={dcg:.3f}, IDCG={idcg:.3f}). "
                f"Higher means relevant chunks are ranked near the top."
            ),
            violations=(
                ["Relevant chunks are not ordered by relevance."] if ndcg < 0.8 else []
            ),
            metadata={"grades": grades, "dcg": dcg, "idcg": idcg, "ndcg": ndcg},
            metric_name=self.name,
        )
