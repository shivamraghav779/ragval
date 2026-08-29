"""Context relevance: per-chunk relevance of each retrieved chunk to the query.

Unlike context precision this is NOT rank-aware and does not consider the
answer. Each chunk is scored independently against the question on a 0-3 scale.
"""

from __future__ import annotations

import asyncio
from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.utils import prompts


class ContextRelevanceMetric(BaseMetric):
    name = "context_relevance"
    description = "Per-chunk relevance of each retrieved chunk to the query"
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

        async def score_chunk(chunk: str):
            prompt = prompts.CONTEXT_RELEVANCE_PROMPT.format(
                question=question, chunk=chunk
            )
            return await provider.complete_json(prompt)

        responses = await asyncio.gather(
            *(score_chunk(c) for c in contexts), return_exceptions=True
        )

        per_chunk = []
        normalized_scores: List[float] = []
        for i, resp in enumerate(responses):
            if isinstance(resp, Exception):
                raw = 0
                reason = f"scoring failed: {resp}"
            else:
                try:
                    raw = max(0, min(3, int(round(float(resp.get("score", 0))))))
                except (TypeError, ValueError):
                    raw = 0
                reason = resp.get("reason", "")
            norm = raw / 3.0
            normalized_scores.append(norm)
            per_chunk.append(
                {
                    "chunk_index": i,
                    "raw_score": raw,
                    "normalized": norm,
                    "reason": reason,
                }
            )

        mean_relevance = sum(normalized_scores) / len(normalized_scores)
        return MetricResult(
            score=mean_relevance,
            reasoning=(
                f"Mean per-chunk relevance {mean_relevance:.3f} across "
                f"{len(contexts)} chunks (0-3 scale, normalized)."
            ),
            violations=[
                f"chunk[{c['chunk_index']}] low relevance ({c['raw_score']}/3)"
                for c in per_chunk
                if c["raw_score"] <= 1
            ],
            metadata={
                "per_chunk_scores": per_chunk,
                "mean_relevance": mean_relevance,
            },
            metric_name=self.name,
        )
