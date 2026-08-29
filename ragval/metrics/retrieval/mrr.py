"""Mean Reciprocal Rank of the first relevant retrieved chunk.

Most meaningful over a batch of queries; for a single query it returns
1 / rank_of_first_relevant.
"""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.utils import prompts
from ragval.utils.scoring import compute_mrr


async def judge_relevance_labels(
    question: str, contexts: List[str], provider: Any
) -> List[bool]:
    """Ask the LLM for a binary relevance label per chunk, in index order."""
    data = await provider.complete_json(
        prompts.CHUNK_RELEVANCE_JUDGE_PROMPT.format(
            question=question, contexts=prompts.join_contexts(contexts)
        )
    )
    labels = [False] * len(contexts)
    for entry in data.get("labels", []) or []:
        try:
            idx = int(entry.get("index"))
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(contexts):
            labels[idx] = bool(entry.get("relevant"))
    return labels


class MRRMetric(BaseMetric):
    name = "mrr"
    description = "Mean Reciprocal Rank of first relevant chunk"
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

        labels = kwargs.get("chunk_relevance_labels")
        if labels is None or len(labels) != len(contexts):
            labels = await judge_relevance_labels(question, contexts, provider)
        labels = [bool(x) for x in labels]

        mrr_value = compute_mrr(labels)
        first_pos = next((i for i, r in enumerate(labels) if r), None)

        return MetricResult(
            score=mrr_value,
            reasoning=(
                f"First relevant chunk at position {first_pos}. MRR = {mrr_value:.3f}."
                if first_pos is not None
                else "No relevant chunk found. MRR = 0.0."
            ),
            violations=[] if first_pos == 0 else ["Top-ranked chunk is not relevant."]
            if first_pos is not None
            else ["No relevant chunk retrieved."],
            metadata={
                "first_relevant_position": first_pos,
                "relevance_labels": labels,
                "mrr_value": mrr_value,
            },
            metric_name=self.name,
        )
