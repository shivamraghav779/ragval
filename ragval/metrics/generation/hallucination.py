"""Hallucination: does the answer contradict or fabricate specific facts?

Distinct from faithfulness. Faithfulness asks "are claims supported?".
Hallucination asks "are specific values (numbers, names, dosages, dates)
contradicted or invented?". A detected hallucination is a hard FAIL for
ragval verdicts.
"""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.utils import prompts, text


class HallucinationMetric(BaseMetric):
    name = "hallucination"
    description = "Does answer contradict or fabricate entities from context?"
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
        seed = text.extract_entities(answer)
        seed_text = (
            prompts.numbered_list([str(e["text"]) for e in seed]) if seed else "(none)"
        )
        extracted = await provider.complete_json(
            prompts.HALLUCINATION_ENTITY_EXTRACT_PROMPT.format(
                answer=answer, seed_entities=seed_text
            )
        )
        entities = []
        for e in extracted.get("entities", []) or []:
            t = (e.get("text") or "").strip()
            if t:
                entities.append({"text": t, "type": e.get("type", "unknown")})

        if not entities:
            return MetricResult(
                score=0.0,
                reasoning=(
                    "No specific checkable facts in the answer; cannot assess "
                    "hallucination."
                ),
                violations=[],
                metadata={
                    "entities": [],
                    "checks": [],
                    "hallucination_score": 0.0,
                    "hallucination_detected": False,
                    "contradicted_entities": [],
                },
                metric_name=self.name,
            )

        check_data = await provider.complete_json(
            prompts.HALLUCINATION_CONTRADICTION_PROMPT.format(
                contexts=prompts.join_contexts(contexts),
                entities=prompts.numbered_list([e["text"] for e in entities]),
            )
        )
        checks = check_data.get("checks", []) or []

        contradicted = []
        for i, ent in enumerate(entities):
            entry = checks[i] if i < len(checks) else {}
            if entry.get("contradicted"):
                detail = entry.get("contradiction_detail") or f"{ent['text']}"
                contradicted.append(detail)

        hallucination_score = len(contradicted) / len(entities)
        hallucination_detected = hallucination_score > 0.1 or bool(contradicted)
        score = 1.0 - hallucination_score

        return MetricResult(
            score=score,
            reasoning=(
                f"{len(contradicted)}/{len(entities)} specific facts contradicted "
                f"by context. hallucination_detected={hallucination_detected}."
            ),
            violations=contradicted,
            metadata={
                "entities": entities,
                "checks": checks,
                "hallucination_score": hallucination_score,
                "hallucination_detected": hallucination_detected,
                "contradicted_entities": contradicted,
            },
            metric_name=self.name,
        )
