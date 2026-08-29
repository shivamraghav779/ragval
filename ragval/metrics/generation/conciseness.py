"""Conciseness: is the answer free of padding, repetition, and filler?"""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.utils import prompts
from ragval.utils.scoring import clamp


class ConcisenessMetric(BaseMetric):
    name = "conciseness"
    description = "Answer is free of redundancy, filler, and padding"
    requires_ground_truth = False
    category = "generation"

    async def _compute(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        provider: Any,
        ground_truth: Optional[str] = None,
        **kwargs: Any,
    ) -> MetricResult:
        if not answer or not answer.strip():
            return self._na("no answer to assess")

        data = await provider.complete_json(
            prompts.CONCISENESS_PROMPT.format(question=question, answer=answer)
        )
        score = clamp(float(data.get("conciseness_score", 1.0) or 0.0))
        redundant = data.get("redundant_spans", []) or []

        return MetricResult(
            score=score,
            reasoning=data.get("reasoning", "") or f"Conciseness {score:.3f}.",
            violations=[f"padding: {r}" for r in redundant],
            metadata={"redundant_spans": redundant},
            metric_name=self.name,
        )
