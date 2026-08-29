"""Hit rate: did at least one relevant chunk appear in the top K?"""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.metrics.retrieval.mrr import judge_relevance_labels


class HitRateMetric(BaseMetric):
    name = "hit_rate"
    description = "Did at least one relevant chunk appear in top K?"
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
            return self._na("no contexts to check")

        k = kwargs.get("k") or len(contexts)
        k = max(1, min(int(k), len(contexts)))

        labels = kwargs.get("chunk_relevance_labels")
        if labels is None or len(labels) != len(contexts):
            labels = await judge_relevance_labels(question, contexts, provider)
        labels = [bool(x) for x in labels]

        top_k = labels[:k]
        relevant_found = any(top_k)
        first_pos = next((i for i, r in enumerate(labels) if r), None)

        return MetricResult(
            score=1.0 if relevant_found else 0.0,
            reasoning=(
                f"{'A' if relevant_found else 'No'} relevant chunk in the top {k}."
            ),
            violations=[] if relevant_found else [f"No relevant chunk in top {k}."],
            metadata={
                "k": k,
                "relevant_found": relevant_found,
                "first_relevant_position": first_pos,
            },
            metric_name=self.name,
        )
