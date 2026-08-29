"""Argument correctness: did the agent pass the right arguments to its tools?"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.utils import prompts


def _index_by_name(calls: List[Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for c in calls or []:
        if isinstance(c, dict):
            name = c.get("name") or c.get("tool") or c.get("function")
            if name:
                out[name] = c.get("arguments", c.get("args", {}))
    return out


class ArgumentCorrectnessMetric(BaseMetric):
    name = "argument_correctness"
    description = "Did agent pass correct arguments to tools?"
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
        expected = kwargs.get("expected_tool_calls")
        if actual is None or expected is None:
            return self._na(
                "argument_correctness needs tool_calls and expected_tool_calls"
            )

        actual_args = _index_by_name(actual)
        expected_args = _index_by_name(expected)
        matched = sorted(set(actual_args) & set(expected_args))
        if not matched:
            return self._na("no tools appear in both actual and expected calls")

        pairs = "\n".join(
            f"- tool '{name}': actual={json.dumps(actual_args[name], default=str)} "
            f"expected={json.dumps(expected_args[name], default=str)}"
            for name in matched
        )
        data = await provider.complete_json(
            prompts.ARGUMENT_CORRECTNESS_PROMPT.format(tool_pairs=pairs)
        )
        verdicts = data.get("tool_argument_verdicts", []) or []

        correct = sum(1 for v in verdicts if v.get("correct"))
        total = len(verdicts) if verdicts else len(matched)
        score = correct / total if total else 0.0

        violations = []
        for v in verdicts:
            for issue in v.get("issues", []) or []:
                violations.append(f"{v.get('tool', '?')}: {issue}")

        return MetricResult(
            score=score,
            reasoning=f"{correct}/{total} matched tools received correct arguments.",
            violations=violations,
            metadata={"tool_argument_verdicts": verdicts},
            metric_name=self.name,
        )
