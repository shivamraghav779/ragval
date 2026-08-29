"""Financial domain: numerical accuracy, regulatory-mention, temporal accuracy."""

from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List

from ragval.domains.base import BaseDomain
from ragval.metrics.base import MetricResult
from ragval.utils import prompts, text

_TIME_SENSITIVE_RE = re.compile(
    r"(interest rate|repo rate|policy rate|market cap|share price|index level|"
    r"yield|inflation rate|exchange rate)",
    re.IGNORECASE,
)
_DATE_HINT_RE = re.compile(
    r"(\b\d{4}\b|\bQ[1-4]\b|as of|current|latest|fiscal year|FY\d{2,4})",
    re.IGNORECASE,
)


class FinancialDomain(BaseDomain):
    name = "financial"
    description = "Financial analysis assistant RAG"
    additional_metric_names = [
        "numerical_accuracy",
        "regulatory_compliance_mention",
        "temporal_accuracy",
    ]
    system_prompt_addition = (
        "You are evaluating a financial analysis assistant used by finance "
        "professionals. Numerical accuracy is critical - wrong figures cause "
        "significant harm. Regulatory compliance context matters. Flag any "
        "specific financial figure that cannot be traced to retrieved context."
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
                "numerical_accuracy", self._numerical(answer, contexts, provider)
            ),
            self._safe(
                "regulatory_compliance_mention",
                self._regulatory(question, answer, provider),
            ),
            self._safe("temporal_accuracy", self._temporal(answer, contexts)),
        )
        return {r.metric_name: r for r in results}

    async def _numerical(
        self, answer: str, contexts: List[str], provider: Any
    ) -> MetricResult:
        seed = [
            str(e["text"])
            for e in text.extract_entities(answer)
            if e["type"] in ("numbers_with_units", "percentage", "currency")
        ]
        extract = await provider.complete_json(
            prompts.FINANCIAL_NUMBER_EXTRACT_PROMPT.format(
                answer=answer,
                seed_entities=prompts.numbered_list(seed) if seed else "(none)",
            )
        )
        numbers = [n.get("text") for n in extract.get("numbers", []) if n.get("text")]
        if not numbers:
            return MetricResult.not_applicable(
                "numerical_accuracy", "no specific financial figures in the answer"
            )
        joined = "\n".join(contexts).lower()
        verified = [n for n in numbers if str(n).lower().strip() in joined]
        score = len(verified) / len(numbers)
        violations = [f"untraceable figure: {n}" for n in numbers if n not in verified]
        if score < 0.8:
            violations.append("financial_accuracy_concern: numerical accuracy below 0.8")
        return MetricResult(
            score=score,
            reasoning=f"{len(verified)}/{len(numbers)} figures traceable to context.",
            violations=violations,
            metadata={"numbers": numbers, "verified": verified},
            metric_name="numerical_accuracy",
        )

    async def _regulatory(
        self, question: str, answer: str, provider: Any
    ) -> MetricResult:
        data = await provider.complete_json(
            prompts.REGULATORY_MENTION_PROMPT.format(question=question, answer=answer)
        )
        if not data.get("requires_regulatory_context"):
            return MetricResult.not_applicable(
                "regulatory_compliance_mention",
                "the query is general, not product/advice specific",
            )
        present = bool(data.get("regulatory_context_present"))
        return MetricResult(
            score=1.0 if present else 0.0,
            reasoning=data.get("reasoning", "")
            or ("Regulatory context present." if present else "Missing regulatory context."),
            violations=[] if present else ["Answer omits required regulatory context."],
            metadata={"regulators_mentioned": data.get("regulators_mentioned", [])},
            metric_name="regulatory_compliance_mention",
        )

    async def _temporal(self, answer: str, contexts: List[str]) -> MetricResult:
        if not _TIME_SENSITIVE_RE.search(answer):
            return MetricResult.not_applicable(
                "temporal_accuracy", "answer contains no time-sensitive data"
            )
        context_has_date = bool(_DATE_HINT_RE.search("\n".join(contexts)))
        if context_has_date:
            score, reasoning = 1.0, "Time-sensitive data is dated in the context."
        else:
            score, reasoning = (
                0.5,
                "Time-sensitive data present but context has no clear date; may be stale.",
            )
        return MetricResult(
            score=score,
            reasoning=reasoning,
            violations=[] if score >= 1.0 else ["Time-sensitive data may be stale."],
            metadata={"context_has_date": context_has_date},
            metric_name="temporal_accuracy",
        )
