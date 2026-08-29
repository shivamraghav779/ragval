"""Clinical domain: drug-name precision, dosing accuracy, contraindication
coverage, and source authority scoring."""

from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List

from ragval.domains.base import BaseDomain
from ragval.metrics.base import MetricResult
from ragval.utils import prompts

_DOSING_RE = re.compile(
    r"\b\d+\.?\d*\s?(?:mg|mcg|g|ml|units?|iu)\b(?:[^.]{0,40})?",
    re.IGNORECASE,
)

_AUTHORITY_SIGNALS = [
    (("who", "world health organization"), 1.0),
    (("cdc", "centers for disease control"), 1.0),
    (("icmr", "nice", "fda"), 0.9),
    (("peer-reviewed", "journal", "pubmed"), 0.7),
    (("hospital protocol",), 0.5),
]


class ClinicalDomain(BaseDomain):
    name = "clinical"
    description = "Clinical decision support RAG"
    additional_metric_names = [
        "drug_name_precision",
        "dosing_accuracy",
        "contraindication_coverage",
        "authority_score",
    ]
    system_prompt_addition = (
        "You are evaluating a clinical decision support system used by "
        "healthcare professionals. Precision in drug names and dosages is "
        "critical for patient safety. Outdated guidelines are as dangerous as "
        "missing ones. Flag any recommendation that could harm a patient even "
        "if technically grounded in retrieved context."
    )

    async def get_domain_metrics(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        provider: Any,
    ) -> Dict[str, MetricResult]:
        results = await asyncio.gather(
            self._safe("drug_name_precision", self._drug_name_precision(answer, contexts, provider)),
            self._safe("dosing_accuracy", self._dosing_accuracy(answer, contexts, provider)),
            self._safe(
                "contraindication_coverage",
                self._contraindication_coverage(answer, contexts, provider),
            ),
            self._safe("authority_score", self._authority_score(contexts)),
        )
        return {r.metric_name: r for r in results}

    async def _drug_name_precision(
        self, answer: str, contexts: List[str], provider: Any
    ) -> MetricResult:
        extract = await provider.complete_json(
            prompts.DRUG_NAME_EXTRACT_PROMPT.format(answer=answer)
        )
        drugs = [d for d in extract.get("drug_names", []) if d]
        if not drugs:
            return MetricResult.not_applicable(
                "drug_name_precision", "no drug names in the answer"
            )
        verify = await provider.complete_json(
            prompts.DRUG_NAME_VERIFY_PROMPT.format(
                contexts=prompts.join_contexts(contexts),
                drug_names=prompts.numbered_list(drugs),
            )
        )
        verdicts = verify.get("verdicts", []) or []
        verified = sum(1 for v in verdicts if v.get("verified"))
        score = verified / len(drugs)
        violations = [
            f"unverified drug name: {v.get('drug')}"
            for v in verdicts
            if not v.get("verified")
        ]
        if score < 0.9:
            violations.append("clinical_safety_concern: drug name precision below 0.9")
        return MetricResult(
            score=score,
            reasoning=f"{verified}/{len(drugs)} drug names verified against context.",
            violations=violations,
            metadata={"drugs": drugs, "verdicts": verdicts},
            metric_name="drug_name_precision",
        )

    async def _dosing_accuracy(
        self, answer: str, contexts: List[str], provider: Any
    ) -> MetricResult:
        doses = [m.group(0).strip() for m in _DOSING_RE.finditer(answer)]
        if not doses:
            return MetricResult.not_applicable(
                "dosing_accuracy", "no dosing information in the answer"
            )
        verify = await provider.complete_json(
            prompts.DOSING_VERIFY_PROMPT.format(
                contexts=prompts.join_contexts(contexts),
                dosing_statements=prompts.numbered_list(doses),
            )
        )
        verdicts = verify.get("verdicts", []) or []
        verified = sum(1 for v in verdicts if v.get("verified"))
        score = verified / len(doses)
        return MetricResult(
            score=score,
            reasoning=f"{verified}/{len(doses)} dosing statements supported by context.",
            violations=[
                f"unverified dose: {v.get('dosing')}"
                for v in verdicts
                if not v.get("verified")
            ],
            metadata={"doses": doses, "verdicts": verdicts},
            metric_name="dosing_accuracy",
        )

    async def _contraindication_coverage(
        self, answer: str, contexts: List[str], provider: Any
    ) -> MetricResult:
        extract = await provider.complete_json(
            prompts.CONTRAINDICATION_EXTRACT_PROMPT.format(
                contexts=prompts.join_contexts(contexts)
            )
        )
        contras = [c for c in extract.get("contraindications", []) if c]
        if not contras:
            return MetricResult.not_applicable(
                "contraindication_coverage",
                "no contraindications present in the retrieved context",
            )
        cov = await provider.complete_json(
            prompts.CONTRAINDICATION_COVERAGE_PROMPT.format(
                answer=answer,
                contraindications=prompts.numbered_list(contras),
            )
        )
        coverage = cov.get("coverage", []) or []
        mentioned = sum(1 for c in coverage if c.get("mentioned"))
        score = mentioned / len(contras)
        return MetricResult(
            score=score,
            reasoning=f"{mentioned}/{len(contras)} context contraindications addressed.",
            violations=[
                f"unaddressed contraindication: {c.get('contraindication')}"
                for c in coverage
                if not c.get("mentioned")
            ],
            metadata={"contraindications": contras, "coverage": coverage},
            metric_name="contraindication_coverage",
        )

    async def _authority_score(self, contexts: List[str]) -> MetricResult:
        joined = "\n".join(contexts).lower()
        levels: List[float] = []
        for signals, weight in _AUTHORITY_SIGNALS:
            if any(s in joined for s in signals):
                levels.append(weight)
        if not levels:
            score = 0.5 if joined.strip() else 0.5
            reasoning = "No recognizable authority signals in the sources."
        else:
            score = sum(levels) / len(levels)
            reasoning = f"Authority signals found; mean level {score:.2f}."
        return MetricResult(
            score=score,
            reasoning=reasoning,
            violations=["Low source authority."] if score < 0.5 else [],
            metadata={"levels": levels},
            metric_name="authority_score",
        )
