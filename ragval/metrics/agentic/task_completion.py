"""Task completion: did the agent achieve the user's goal?"""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.utils import prompts
from ragval.utils.scoring import clamp


def format_trace(trace: List[Any]) -> str:
    if not trace:
        return "(no action trace provided)"
    lines = []
    for i, step in enumerate(trace):
        if isinstance(step, dict):
            desc = step.get("description") or step.get("action") or str(step)
        else:
            desc = str(step)
        lines.append(f"{i + 1}. {desc}")
    return "\n".join(lines)


class TaskCompletionMetric(BaseMetric):
    name = "task_completion"
    description = "Did agent achieve the user goal?"
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
        action_trace = kwargs.get("action_trace", []) or []

        data = await provider.complete_json(
            prompts.TASK_COMPLETION_PROMPT.format(
                question=question,
                answer=answer,
                action_trace=format_trace(action_trace),
            )
        )
        score = clamp(float(data.get("completion_score", 0.0) or 0.0))
        unmet = data.get("unmet_requirements", []) or []

        return MetricResult(
            score=score,
            reasoning=data.get("reasoning", "") or f"Completion score {score:.3f}.",
            violations=[f"unmet: {u}" for u in unmet],
            metadata={
                "task_completed": bool(data.get("task_completed")),
                "action_trace_length": len(action_trace),
                "unmet_requirements": unmet,
            },
            metric_name=self.name,
        )
