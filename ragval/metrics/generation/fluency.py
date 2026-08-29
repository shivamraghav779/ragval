"""Fluency: grammar, spelling, punctuation, and natural readability (form only)."""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.utils import prompts


class FluencyMetric(BaseMetric):
    name = "fluency"
    description = "Grammatical correctness and readability of the answer"
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
            prompts.FLUENCY_PROMPT.format(answer=answer)
        )
        try:
            raw = max(1, min(5, int(round(float(data.get("score", 3))))))
        except (TypeError, ValueError):
            raw = 3
        score = (raw - 1) / 4.0
        issues = data.get("issues", []) or []

        return MetricResult(
            score=score,
            reasoning=data.get("reasoning", "") or f"Fluency {raw}/5.",
            violations=issues,
            metadata={"raw_score": raw, "issues": issues},
            metric_name=self.name,
        )
