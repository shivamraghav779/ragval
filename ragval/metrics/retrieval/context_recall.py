"""Context recall: does retrieval cover everything needed to answer?"""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.utils import prompts


class ContextRecallMetric(BaseMetric):
    name = "context_recall"
    description = "Coverage of information needed to answer"
    requires_ground_truth = True
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
        if ground_truth is None:
            return self._na("context_recall requires ground_truth")
        if not contexts:
            return MetricResult(
                score=0.0,
                reasoning="No context retrieved, so nothing can be recalled.",
                violations=["empty context"],
                metadata={"statements": [], "attributions": []},
                metric_name=self.name,
            )

        extract = await provider.complete_json(
            prompts.CONTEXT_RECALL_EXTRACT_PROMPT.format(ground_truth=ground_truth)
        )
        statements = [s for s in extract.get("statements", []) if s]
        if not statements:
            return self._na("could not extract statements from ground_truth")

        verify = await provider.complete_json(
            prompts.CONTEXT_RECALL_VERIFY_PROMPT.format(
                contexts=prompts.join_contexts(contexts),
                statements=prompts.numbered_list(statements),
            )
        )
        attributions = verify.get("attributions", []) or []

        attributed = 0
        unattributed: List[str] = []
        for i, stmt in enumerate(statements):
            entry = attributions[i] if i < len(attributions) else {}
            if entry.get("attributed"):
                attributed += 1
            else:
                unattributed.append(stmt)

        score = attributed / len(statements)
        return MetricResult(
            score=score,
            reasoning=(
                f"{attributed}/{len(statements)} reference statements are "
                f"supported by the retrieved context."
            ),
            violations=[f"not retrieved: {s}" for s in unattributed],
            metadata={
                "statements": statements,
                "attributions": attributions,
                "attributed_count": attributed,
                "total_statements": len(statements),
            },
            metric_name=self.name,
        )
