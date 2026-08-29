"""Answer completeness: does the answer address every part the question asks?"""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.utils import prompts


class AnswerCompletenessMetric(BaseMetric):
    name = "answer_completeness"
    description = "Does the answer address every distinct part of the question?"
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
            return MetricResult(
                score=0.0,
                reasoning="Empty answer addresses nothing.",
                violations=["empty answer"],
                metadata={"requirements": []},
                metric_name=self.name,
            )

        data = await provider.complete_json(
            prompts.ANSWER_COMPLETENESS_PROMPT.format(question=question, answer=answer)
        )
        requirements = data.get("requirements", []) or []
        if not requirements:
            return self._na("could not decompose the question into parts")

        addressed = sum(1 for r in requirements if r.get("addressed"))
        score = addressed / len(requirements)

        return MetricResult(
            score=score,
            reasoning=(
                f"{addressed}/{len(requirements)} distinct information needs in "
                f"the question are addressed by the answer."
            ),
            violations=[
                f"unaddressed: {r.get('need', '')}"
                for r in requirements
                if not r.get("addressed")
            ],
            metadata={
                "requirements": requirements,
                "addressed_count": addressed,
                "total_requirements": len(requirements),
            },
            metric_name=self.name,
        )
