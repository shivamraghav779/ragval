"""Legal domain: jurisdiction specificity, citation accuracy, statute currency."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from ragval.domains.base import BaseDomain
from ragval.metrics.base import MetricResult
from ragval.utils import prompts
from ragval.utils.scoring import clamp

_JURISDICTION_SCORE = {"clearly": 1.0, "implied": 0.5, "missing": 0.0}


class LegalDomain(BaseDomain):
    name = "legal"
    description = "Legal research assistant RAG"
    additional_metric_names = [
        "jurisdiction_specificity",
        "citation_accuracy",
        "statute_currency",
    ]
    system_prompt_addition = (
        "You are evaluating a legal research assistant used by legal "
        "professionals. Jurisdiction specificity is critical - a law applicable "
        "in one jurisdiction may not apply in another. Statute versions matter - "
        "superseded legislation cited as current is a serious error. Flag "
        "jurisdiction omissions when the question implies jurisdiction."
    )

    async def get_domain_metrics(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        provider: Any,
    ) -> Dict[str, MetricResult]:
        results = await asyncio.gather(
            self._safe(
                "jurisdiction_specificity",
                self._jurisdiction(question, answer, provider),
            ),
            self._safe(
                "citation_accuracy", self._citations(answer, contexts, provider)
            ),
            self._safe(
                "statute_currency", self._currency(answer, contexts, provider)
            ),
        )
        return {r.metric_name: r for r in results}

    async def _jurisdiction(
        self, question: str, answer: str, provider: Any
    ) -> MetricResult:
        data = await provider.complete_json(
            prompts.JURISDICTION_DETECTION_PROMPT.format(
                question=question, answer=answer
            )
        )
        if not data.get("question_implies_jurisdiction"):
            return MetricResult.not_applicable(
                "jurisdiction_specificity",
                "the question does not imply a specific jurisdiction",
            )
        state = str(data.get("answer_specifies_jurisdiction", "missing")).lower()
        score = _JURISDICTION_SCORE.get(state, 0.0)
        return MetricResult(
            score=score,
            reasoning=data.get("reasoning", "") or f"Jurisdiction is {state}.",
            violations=(
                ["Answer omits the jurisdiction the question requires."]
                if score < 1.0
                else []
            ),
            metadata={"jurisdiction_state": state},
            metric_name="jurisdiction_specificity",
        )

    async def _citations(
        self, answer: str, contexts: List[str], provider: Any
    ) -> MetricResult:
        extract = await provider.complete_json(
            prompts.LEGAL_CITATION_EXTRACT_PROMPT.format(answer=answer)
        )
        citations = [c for c in extract.get("citations", []) if c]
        if not citations:
            return MetricResult.not_applicable(
                "citation_accuracy", "no legal citations in the answer"
            )
        joined = "\n".join(contexts).lower()
        verified = [c for c in citations if c.lower() in joined]
        missing = [c for c in citations if c.lower() not in joined]
        score = len(verified) / len(citations)
        return MetricResult(
            score=score,
            reasoning=f"{len(verified)}/{len(citations)} citations appear in context.",
            violations=[f"unverified citation: {c}" for c in missing],
            metadata={"citations": citations, "verified": verified, "missing": missing},
            metric_name="citation_accuracy",
        )

    async def _currency(
        self, answer: str, contexts: List[str], provider: Any
    ) -> MetricResult:
        data = await provider.complete_json(
            prompts.STATUTE_CURRENCY_PROMPT.format(
                answer=answer, contexts=prompts.join_contexts(contexts)
            )
        )
        score = clamp(float(data.get("currency_score", 1.0) or 0.0))
        outdated = data.get("potentially_outdated", []) or []
        return MetricResult(
            score=score,
            reasoning=data.get("reasoning", "") or f"Statute currency {score:.3f}.",
            violations=[f"potentially outdated: {o}" for o in outdated],
            metadata={"potentially_outdated": outdated},
            metric_name="statute_currency",
        )
