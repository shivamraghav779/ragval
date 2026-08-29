"""Refusal appropriateness: was the decision to answer (or decline) correct?

Rewards a system that answers when the context supports an answer and declines
when it does not. Penalizes both over-refusal (context had the answer) and
under-refusal (answering from thin air).
"""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.utils import prompts


class RefusalAppropriatenessMetric(BaseMetric):
    name = "refusal_appropriateness"
    description = "Did the system correctly choose to answer vs decline?"
    requires_ground_truth = False
    category = "generation"

    async def _compute(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        provider: Any,
        ground_truth: Optional[str] = None,
        **kwargs: Any,
    ) -> MetricResult:
        if not answer or not answer.strip():
            return self._na("no answer to assess")

        data = await provider.complete_json(
            prompts.REFUSAL_APPROPRIATENESS_PROMPT.format(
                question=question,
                answer=answer,
                contexts=prompts.join_contexts(contexts),
            )
        )
        did_refuse = bool(data.get("did_refuse"))
        supported = bool(data.get("context_supports_answer"))
        appropriate = bool(data.get("appropriate"))

        if appropriate:
            violations: List[str] = []
        elif did_refuse:
            violations = ["Over-refusal: the context did support an answer."]
        else:
            violations = ["Under-refusal: answered without support in the context."]

        return MetricResult(
            score=1.0 if appropriate else 0.0,
            reasoning=data.get("reasoning", "")
            or (
                f"did_refuse={did_refuse}, context_supports_answer={supported} -> "
                f"{'appropriate' if appropriate else 'inappropriate'}."
            ),
            violations=violations,
            metadata={
                "did_refuse": did_refuse,
                "context_supports_answer": supported,
                "appropriate": appropriate,
            },
            metric_name=self.name,
        )
