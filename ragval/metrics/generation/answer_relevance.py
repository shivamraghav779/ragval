"""Answer relevance: does the answer actually address the question asked?

Method: generate questions the answer would be a good response to, then measure
their similarity to the original question. Low similarity means the answer
drifted from what was asked.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.utils import prompts, text

_REFUSAL_RE = re.compile(
    r"(i cannot find|not in (the )?context|i don't have|no information|"
    r"cannot answer|unable to answer)",
    re.IGNORECASE,
)


class AnswerRelevanceMetric(BaseMetric):
    name = "answer_relevance"
    description = "Does the answer address the question asked?"
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
                reasoning="Empty answer cannot be relevant.",
                violations=["empty answer"],
                metadata={"generated_questions": [], "similarities": [],
                          "is_refusal": False},
                metric_name=self.name,
            )

        is_refusal = bool(_REFUSAL_RE.search(answer))

        gen = await provider.complete_json(
            prompts.ANSWER_RELEVANCE_GENERATE_PROMPT.format(answer=answer)
        )
        generated = [q for q in gen.get("questions", []) if q][:3]
        if not generated:
            return self._na("model produced no candidate questions")

        similarities = [
            text.sentence_similarity(question, gq) for gq in generated
        ]
        score = sum(similarities) / len(similarities)

        return MetricResult(
            score=score,
            reasoning=(
                f"Mean similarity {score:.3f} between the original question and "
                f"{len(generated)} questions reconstructed from the answer."
                + (" Answer appears to be a refusal." if is_refusal else "")
            ),
            violations=(
                ["Answer does not address the question asked."] if score < 0.3 else []
            ),
            metadata={
                "generated_questions": generated,
                "similarities": similarities,
                "is_refusal": is_refusal,
            },
            metric_name=self.name,
        )
