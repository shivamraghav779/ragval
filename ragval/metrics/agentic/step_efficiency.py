"""Step efficiency: did the agent use the minimum necessary steps?"""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.agentic.task_completion import format_trace
from ragval.metrics.base import BaseMetric, MetricResult
from ragval.utils import prompts
from ragval.utils.scoring import clamp


class StepEfficiencyMetric(BaseMetric):
    name = "step_efficiency"
    description = "Did agent use minimum necessary steps?"
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
        action_trace = kwargs.get("action_trace")
        if not action_trace:
            return self._na("step_efficiency needs an action_trace")

        actual_steps = len(action_trace)
        expected_min = kwargs.get("expected_min_steps")

        if expected_min is not None:
            expected_min = int(expected_min)
            score = min(1.0, expected_min / actual_steps) if actual_steps else 0.0
            redundant: List[str] = []
            reasoning = (
                f"{actual_steps} steps taken vs {expected_min} minimum. "
                f"Efficiency = {score:.3f}."
            )
        else:
            data = await provider.complete_json(
                prompts.STEP_EFFICIENCY_PROMPT.format(
                    question=question, action_trace=format_trace(action_trace)
                )
            )
            score = clamp(float(data.get("efficiency_score", 0.0) or 0.0))
            redundant = data.get("redundant_steps", []) or []
            expected_min = data.get("min_steps_estimate")
            reasoning = (
                f"{actual_steps} steps taken; model estimates {expected_min} needed. "
                f"Efficiency = {score:.3f}."
            )

        return MetricResult(
            score=score,
            reasoning=reasoning,
            violations=[f"redundant step: {r}" for r in redundant],
            metadata={
                "actual_steps": actual_steps,
                "expected_min_steps": expected_min,
                "redundant_steps": redundant,
                "efficiency_ratio": (
                    expected_min / actual_steps
                    if expected_min and actual_steps
                    else None
                ),
            },
            metric_name=self.name,
        )
