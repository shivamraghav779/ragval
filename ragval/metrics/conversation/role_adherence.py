"""Role adherence: does the assistant stay in its defined role?"""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.utils import prompts
from ragval.utils.scoring import clamp


class RoleAdherenceMetric(BaseMetric):
    name = "role_adherence"
    description = "Does agent stay in its defined role?"
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
        system_role = kwargs.get("system_role")
        if system_role is None:
            return self._na("role_adherence requires system_role")

        assistant_turns = [
            (i, t) for i, t in enumerate(turns) if t.get("role") == "assistant"
        ]
        if not assistant_turns:
            return self._na("no assistant turns to evaluate")

        rendered = "\n".join(
            f"[turn {i}] {t.get('content', '')}" for i, t in assistant_turns
        )
        data = await provider.complete_json(
            prompts.ROLE_ADHERENCE_PROMPT.format(
                system_role=system_role, turns=rendered
            )
        )
        verdicts = data.get("adherence_verdicts", []) or []
        overall = clamp(float(data.get("overall_adherence", 1.0) or 0.0))

        violations = [
            f"turn {v.get('turn_index')}: {v.get('violation_type', 'off-role')}"
            for v in verdicts
            if not v.get("adheres")
        ]

        return MetricResult(
            score=overall,
            reasoning=f"Overall role adherence {overall:.3f} across "
            f"{len(assistant_turns)} assistant turns.",
            violations=violations,
            metadata={
                "system_role": system_role,
                "adherence_verdicts": verdicts,
                "assistant_turn_count": len(assistant_turns),
            },
            metric_name=self.name,
        )
