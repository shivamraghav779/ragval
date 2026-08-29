"""Context utilization: rank-aware precision judged against the ANSWER.

Where ``context_precision`` judges each chunk's relevance to the question (and
can use a reference answer), this reference-free variant asks a different
question: did the produced answer actually draw on each retrieved chunk?
"""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.utils import prompts


class ContextUtilizationMetric(BaseMetric):
    name = "context_utilization"
    description = "Rank-aware precision judged by whether the answer used each chunk"
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
        if not answer or not answer.strip():
            return self._na("no answer to trace chunk usage against")

        data = await provider.complete_json(
            prompts.CONTEXT_UTILIZATION_PROMPT.format(
                question=question,
                answer=answer,
                contexts=prompts.join_contexts(contexts),
            )
        )
        used = [False] * len(contexts)
        verdicts = []
        for v in data.get("verdicts", []) or []:
            try:
                idx = int(v.get("index"))
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(contexts):
                used[idx] = bool(v.get("used"))
                verdicts.append(
                    {"index": idx, "used": used[idx], "reason": v.get("reason", "")}
                )

        used_count = sum(used)
        if used_count == 0:
            return MetricResult(
                score=0.0,
                reasoning="The answer does not appear to use any retrieved chunk.",
                violations=["Answer is not grounded in retrieval."],
                metadata={"chunk_verdicts": verdicts, "used_count": 0,
                          "total_chunks": len(contexts), "map_components": []},
                metric_name=self.name,
            )

        map_components = []
        seen = 0
        for k, is_used in enumerate(used, start=1):
            if is_used:
                seen += 1
                map_components.append(seen / k)
        score = sum(map_components) / len(map_components)

        return MetricResult(
            score=score,
            reasoning=(
                f"{used_count}/{len(contexts)} chunks were used by the answer. "
                f"Rank-aware utilization MAP = {score:.3f}."
            ),
            violations=[
                f"chunk[{i}] retrieved but unused" for i, u in enumerate(used) if not u
            ],
            metadata={
                "chunk_verdicts": verdicts,
                "used_count": used_count,
                "total_chunks": len(contexts),
                "map_components": map_components,
            },
            metric_name=self.name,
        )
