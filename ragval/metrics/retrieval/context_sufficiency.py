"""Context sufficiency: do the retrieved chunks contain enough to answer?

Reference-free counterpart to context_recall. Recall needs a ground-truth
answer to know what "complete" means; sufficiency asks the judge directly
whether the context alone is enough to answer the question.
"""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.utils import prompts
from ragval.utils.scoring import clamp


class ContextSufficiencyMetric(BaseMetric):
    name = "context_sufficiency"
    description = "Do retrieved chunks contain enough to fully answer? (reference-free)"
    requires_ground_truth = False
    category = "retrieval"

    async def _compute(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        provider: Any,
        ground_truth: Optional[str] = None,
        **kwargs: Any,
    ) -> MetricResult:
        if not contexts:
            return MetricResult(
                score=0.0,
                reasoning="No context retrieved; cannot be sufficient.",
                violations=["empty context"],
                metadata={"missing_information": ["everything"]},
                metric_name=self.name,
            )

        data = await provider.complete_json(
            prompts.CONTEXT_SUFFICIENCY_PROMPT.format(
                question=question, contexts=prompts.join_contexts(contexts)
            )
        )
        score = clamp(float(data.get("sufficiency_score", 0.0) or 0.0))
        missing = data.get("missing_information", []) or []

        return MetricResult(
            score=score,
            reasoning=data.get("reasoning", "")
            or f"Context sufficiency {score:.3f}.",
            violations=[f"context lacks: {m}" for m in missing],
            metadata={
                "sufficient": bool(data.get("sufficient")),
                "missing_information": missing,
            },
            metric_name=self.name,
        )
