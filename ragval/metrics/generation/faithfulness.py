"""Faithfulness: are all claims in the answer supported by the context?"""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.utils import prompts


class FaithfulnessMetric(BaseMetric):
    name = "faithfulness"
    description = "Are all answer claims supported by context?"
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
                score=1.0,
                reasoning="Empty answer makes no claims; vacuously faithful.",
                violations=[],
                metadata={"claims": [], "verdicts": [], "total_claims": 0,
                          "supported_claims": 0, "unsupported_claims": []},
                metric_name=self.name,
            )

        decomposed = await provider.complete_json(
            prompts.FAITHFULNESS_DECOMPOSE_PROMPT.format(answer=answer)
        )
        claims = [c for c in decomposed.get("claims", []) if c]
        if not claims:
            return MetricResult(
                score=1.0,
                reasoning="No verifiable factual claims found in the answer.",
                violations=[],
                metadata={"claims": [], "verdicts": [], "total_claims": 0,
                          "supported_claims": 0, "unsupported_claims": []},
                metric_name=self.name,
            )

        verify = await provider.complete_json(
            prompts.FAITHFULNESS_VERIFY_PROMPT.format(
                contexts=prompts.join_contexts(contexts),
                claims=prompts.numbered_list(claims),
            )
        )
        verdicts = verify.get("verdicts", []) or []

        supported = 0
        unsupported: List[str] = []
        for i, claim in enumerate(claims):
            entry = verdicts[i] if i < len(verdicts) else {}
            if entry.get("supported"):
                supported += 1
            else:
                unsupported.append(claim)

        score = supported / len(claims)
        return MetricResult(
            score=score,
            reasoning=(
                f"{supported}/{len(claims)} answer claims are grounded in the "
                f"retrieved context."
            ),
            violations=[f"unsupported claim: {c}" for c in unsupported],
            metadata={
                "claims": claims,
                "verdicts": verdicts,
                "total_claims": len(claims),
                "supported_claims": supported,
                "unsupported_claims": unsupported,
            },
            metric_name=self.name,
        )
