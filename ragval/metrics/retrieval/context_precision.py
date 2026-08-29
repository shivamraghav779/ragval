"""Context precision: rank-aware precision of retrieved chunks (MAP)."""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.utils import prompts


class ContextPrecisionMetric(BaseMetric):
    name = "context_precision"
    description = "Rank-aware precision of retrieved chunks"
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
            return self._na("no contexts to score")

        prompt = prompts.CONTEXT_PRECISION_PROMPT.format(
            question=question,
            answer=answer or "(no answer provided)",
            contexts=prompts.join_contexts(contexts),
        )
        data = await provider.complete_json(prompt)
        raw_verdicts = data.get("verdicts", []) or []

        # Normalize to one bool per chunk, in index order.
        relevance = [False] * len(contexts)
        verdicts = []
        for v in raw_verdicts:
            try:
                idx = int(v.get("index"))
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(contexts):
                relevance[idx] = bool(v.get("relevant"))
                verdicts.append(
                    {
                        "index": idx,
                        "relevant": bool(v.get("relevant")),
                        "reason": v.get("reason", ""),
                    }
                )

        relevant_count = sum(relevance)
        if relevant_count == 0:
            return MetricResult(
                score=0.0,
                reasoning="No retrieved chunk was judged relevant to the question.",
                violations=["All retrieved chunks are irrelevant."],
                metadata={
                    "chunk_verdicts": verdicts,
                    "relevant_count": 0,
                    "total_chunks": len(contexts),
                    "map_components": [],
                },
                metric_name=self.name,
            )

        # Mean Average Precision over positions where the chunk is relevant.
        map_components: List[float] = []
        seen_relevant = 0
        for k, is_rel in enumerate(relevance, start=1):
            if is_rel:
                seen_relevant += 1
                map_components.append(seen_relevant / k)
        score = sum(map_components) / len(map_components)

        return MetricResult(
            score=score,
            reasoning=(
                f"{relevant_count}/{len(contexts)} chunks relevant. "
                f"Rank-aware MAP = {score:.3f}."
            ),
            violations=[
                f"chunk[{i}] irrelevant" for i, r in enumerate(relevance) if not r
            ],
            metadata={
                "chunk_verdicts": verdicts,
                "relevant_count": relevant_count,
                "total_chunks": len(contexts),
                "map_components": map_components,
            },
            metric_name=self.name,
        )
