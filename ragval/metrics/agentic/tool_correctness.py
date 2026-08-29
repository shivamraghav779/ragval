"""Tool correctness: did the agent call the right tools? (deterministic, no LLM)"""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult


def _tool_names(calls: List[Any]) -> List[str]:
    names = []
    for c in calls or []:
        if isinstance(c, str):
            names.append(c)
        elif isinstance(c, dict):
            names.append(c.get("name") or c.get("tool") or c.get("function") or "")
        else:
            names.append(str(c))
    return [n for n in names if n]


class ToolCorrectnessMetric(BaseMetric):
    name = "tool_correctness"
    description = "Did agent call the right tools?"
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
        actual = kwargs.get("tool_calls")
        expected = kwargs.get("expected_tools")
        if actual is None or expected is None:
            return self._na("tool_correctness needs tool_calls and expected_tools")

        actual_set = set(_tool_names(actual))
        expected_set = set(_tool_names(expected))

        tp = len(actual_set & expected_set)
        fp = len(actual_set - expected_set)
        fn = len(expected_set - actual_set)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        violations = [
            f"Tool '{t}' called but not expected" for t in sorted(actual_set - expected_set)
        ] + [
            f"Expected tool '{t}' not called" for t in sorted(expected_set - actual_set)
        ]

        return MetricResult(
            score=f1,
            reasoning=(
                f"Tool selection F1={f1:.3f} (precision={precision:.3f}, "
                f"recall={recall:.3f}, TP={tp}, FP={fp}, FN={fn})."
            ),
            violations=violations,
            metadata={
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "unexpected_tools": sorted(actual_set - expected_set),
                "missing_tools": sorted(expected_set - actual_set),
            },
            metric_name=self.name,
        )
