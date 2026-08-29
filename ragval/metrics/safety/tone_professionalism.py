"""Tone professionalism: is the answer's tone appropriate for its audience?"""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.utils import prompts
from ragval.utils.scoring import clamp


class ToneProfessionalismMetric(BaseMetric):
    name = "tone_professionalism"
    description = "Answer tone is professional and audience-appropriate"
    requires_ground_truth = False
    category = "safety"

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
            prompts.TONE_PROFESSIONALISM_PROMPT.format(question=question, answer=answer)
        )
        score = clamp(float(data.get("score", 1.0) or 0.0))
        issues = data.get("tone_issues", []) or []

        return MetricResult(
            score=score,
            reasoning=data.get("reasoning", "") or f"Tone professionalism {score:.3f}.",
            violations=[f"tone: {i}" for i in issues],
            metadata={"tone_issues": issues},
            metric_name=self.name,
        )
