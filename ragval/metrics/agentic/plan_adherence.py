"""Plan adherence: did the agent follow its declared plan?"""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.agentic.task_completion import format_trace
from ragval.metrics.base import BaseMetric, MetricResult
from ragval.utils import prompts
from ragval.utils.scoring import clamp


class PlanAdherenceMetric(BaseMetric):
    name = "plan_adherence"
    description = "Did agent follow its declared plan?"
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
            return self._na("plan_adherence needs a declared_plan")

        action_trace = kwargs.get("action_trace", []) or []
        data = await provider.complete_json(
            prompts.PLAN_ADHERENCE_PROMPT.format(
                declared_plan=declared_plan,
                action_trace=format_trace(action_trace),
            )
        )
        score = clamp(float(data.get("adherence_score", 0.0) or 0.0))
        deviations = data.get("plan_deviations", []) or []

        return MetricResult(
            score=score,
            reasoning=data.get("reasoning", "") or f"Plan adherence {score:.3f}.",
            violations=[f"deviation: {d}" for d in deviations],
            metadata={
                "declared_plan": declared_plan,
                "action_trace_summary": format_trace(action_trace),
            },
            metric_name=self.name,
        )
