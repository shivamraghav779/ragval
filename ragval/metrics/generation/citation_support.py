"""Citation support: are the answer's quoted/attributed spans actually in context?

Distinct from faithfulness (which checks every claim). This targets spans the
answer explicitly presents as coming from the sources — quotes, figures,
"according to..." attributions.
"""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.utils import prompts


class CitationSupportMetric(BaseMetric):
    name = "citation_support"
    description = "Are the answer's attributed/quoted spans present in the context?"
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
        if not contexts:
            return self._na("no context to verify citations against")

        data = await provider.complete_json(
            prompts.CITATION_SUPPORT_PROMPT.format(
                answer=answer, contexts=prompts.join_contexts(contexts)
            )
        )
        citations = data.get("citations", []) or []
        if not citations:
            return self._na("answer makes no specific attributed claims")

        supported = sum(1 for c in citations if c.get("supported"))
        score = supported / len(citations)

        return MetricResult(
            score=score,
            reasoning=(
                f"{supported}/{len(citations)} attributed spans are backed by the "
                f"retrieved context."
            ),
            violations=[
                f"unsupported citation: {c.get('span', '')}"
                for c in citations
                if not c.get("supported")
            ],
            metadata={"citations": citations, "supported_count": supported},
            metric_name=self.name,
        )
