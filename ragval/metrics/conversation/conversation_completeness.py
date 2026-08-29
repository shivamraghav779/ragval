"""Conversation completeness: were all user needs addressed over the dialogue?"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.utils import prompts
from ragval.utils.scoring import clamp


def format_turns(turns: List[Dict[str, Any]]) -> str:
    lines = []
    for i, t in enumerate(turns or []):
        role = t.get("role", "?")
        content = t.get("content", "")
        lines.append(f"[{i}] {role}: {content}")
    return "\n".join(lines) if lines else "(no turns)"


class ConversationCompletenessMetric(BaseMetric):
    name = "conversation_completeness"
    description = "Were all user needs addressed over conversation?"
    requires_ground_truth = False
    category = "conversation"

    async def _compute(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        provider: Any,
        ground_truth: Optional[str] = None,
        **kwargs: Any,
    ) -> MetricResult:
        turns = kwargs.get("turns", []) or []
        if not turns:
            return self._na("conversation_completeness needs turns")

        user_turns = [t for t in turns if t.get("role") == "user"]
        if not user_turns:
            return self._na("no user turns in conversation")

        data = await provider.complete_json(
            prompts.COMPLETENESS_PROMPT.format(turns=format_turns(turns))
        )
        satisfaction = data.get("need_satisfaction", []) or []

        fully = sum(1 for s in satisfaction if s.get("satisfied"))
        partial = sum(
            1
            for s in satisfaction
            if s.get("partially_satisfied") and not s.get("satisfied")
        )
        total = len(user_turns)
        score = clamp((fully + 0.5 * partial) / total) if total else 0.0

        violations = []
        for s in satisfaction:
            if not s.get("satisfied"):
                for gap in s.get("gaps", []) or []:
                    violations.append(f"unmet need: {gap}")

        return MetricResult(
            score=score,
            reasoning=(
                f"{fully} fully and {partial} partially satisfied of {total} user "
                f"turns. Score={score:.3f}."
            ),
            violations=violations,
            metadata={
                "user_turn_count": total,
                "fully_satisfied": fully,
                "partially_satisfied": partial,
                "unsatisfied": total - fully - partial,
            },
            metric_name=self.name,
        )
