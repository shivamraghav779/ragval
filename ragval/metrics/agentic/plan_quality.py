"""Plan quality: was the agent's plan reasonable for the task?"""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.utils import prompts
from ragval.utils.scoring import clamp


class PlanQualityMetric(BaseMetric):
    name = "plan_quality"
    description = "Was the agent plan reasonable for the task?"
    requires_ground_truth = False
    category = "agentic"

    async def _compute(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        provider: Any,
        ground_truth: Optional[str] = None,
        **kwargs: Any,
    ) -> MetricResult:
        declared_plan = kwargs.get("declared_plan")
        if not declared_plan:
            return self._na("plan_quality needs a declared_plan")

        data = await provider.complete_json(
            prompts.PLAN_QUALITY_PROMPT.format(
                question=question, declared_plan=declared_plan
            )
        )
        score = clamp(float(data.get("quality_score", 0.0) or 0.0))
        weaknesses = data.get("weaknesses", []) or []

        return MetricResult(
            score=score,
            reasoning=data.get("reasoning", "") or f"Plan quality {score:.3f}.",
            violations=[f"weakness: {w}" for w in weaknesses],
            metadata={
                "strengths": data.get("strengths", []) or [],
                "weaknesses": weaknesses,
            },
            metric_name=self.name,
        )
