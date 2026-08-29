"""Knowledge retention: does the agent remember and use earlier-turn info?"""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.metrics.conversation.conversation_completeness import format_turns
from ragval.utils import prompts


class KnowledgeRetentionMetric(BaseMetric):
    name = "knowledge_retention"
    description = "Does agent remember and use earlier turn info?"
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
        if len(turns) < 4:
            return self._na("insufficient turns for retention test")

        half = len(turns) // 2
        early, late = turns[:half], turns[half:]

        facts_data = await provider.complete_json(
            prompts.FACT_EXTRACTION_PROMPT.format(turns=format_turns(early))
        )
        facts = [f for f in facts_data.get("established_facts", []) if f]
        if not facts:
            return self._na("no established facts found in early turns")

        check_data = await provider.complete_json(
            prompts.RETENTION_CHECK_PROMPT.format(
                facts=prompts.numbered_list(facts),
                turns=format_turns(late),
            )
        )
        checks = check_data.get("retention_checks", []) or []

        correctly_used = sum(1 for c in checks if c.get("correctly_used"))
        score = correctly_used / len(facts)
        not_retained = [
            c.get("fact", "")
            for c in checks
            if not c.get("correctly_used")
        ]

        return MetricResult(
            score=score,
            reasoning=(
                f"{correctly_used}/{len(facts)} early-turn facts were correctly "
                f"retained and used later."
            ),
            violations=[f"not retained: {f}" for f in not_retained if f],
            metadata={
                "established_facts": facts,
                "retention_checks": checks,
                "turn_count": len(turns),
            },
            metric_name=self.name,
        )
