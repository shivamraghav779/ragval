"""Agent goal accuracy: outcome vs a reference expected outcome.

Unlike task_completion (which judges the trajectory), this judges only the
final answer against ground_truth.
"""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.utils import prompts
from ragval.utils.scoring import clamp


class AgentGoalAccuracyMetric(BaseMetric):
    name = "agent_goal_accuracy"
    description = "Did agent achieve intended goal regardless of steps?"
    requires_ground_truth = True
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
        if ground_truth is None:
            return self._na("agent_goal_accuracy requires ground_truth")

        data = await provider.complete_json(
            prompts.AGENT_GOAL_ACCURACY_PROMPT.format(
                question=question, answer=answer, ground_truth=ground_truth
            )
        )
        score = clamp(float(data.get("accuracy_score", 0.0) or 0.0))
        gaps = data.get("gaps", []) or []

        return MetricResult(
            score=score,
            reasoning=data.get("reasoning", "") or f"Goal accuracy {score:.3f}.",
            violations=[f"gap: {g}" for g in gaps],
            metadata={
                "goal_achieved": bool(data.get("goal_achieved")),
                "expected_outcome_summary": (ground_truth or "")[:300],
            },
            metric_name=self.name,
        )
