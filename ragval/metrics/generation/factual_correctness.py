"""Factual correctness: claim precision and recall reported separately.

Unlike answer_correctness (which blends into one score with a semantic term),
this keeps precision and recall visible so the diagnosis engine can tell
"answer says too much" from "answer says too little".
"""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.utils import prompts
from ragval.utils.scoring import clamp


class FactualCorrectnessMetric(BaseMetric):
    name = "factual_correctness"
    description = "Precision and recall of factual claims separately"
    requires_ground_truth = True
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
        if ground_truth is None:
            return self._na("factual_correctness requires ground_truth")

        answer_claims = [
            c
            for c in (
                await provider.complete_json(
                    prompts.CLAIM_EXTRACT_PROMPT.format(text=answer)
                )
            ).get("claims", [])
            if c
        ]
        gt_claims = [
            c
            for c in (
                await provider.complete_json(
                    prompts.CLAIM_EXTRACT_PROMPT.format(text=ground_truth)
                )
            ).get("claims", [])
            if c
        ]

        correct_in_answer = 0
        if answer_claims:
            fwd = (
                await provider.complete_json(
                    prompts.CLAIM_MATCH_PROMPT.format(
                        reference_claims=prompts.numbered_list(gt_claims),
                        answer_claims=prompts.numbered_list(answer_claims),
                    )
                )
            ).get("matches", []) or []
            correct_in_answer = sum(1 for e in fwd if e.get("matched"))

        covered_in_answer = 0
        if gt_claims:
            rev = (
                await provider.complete_json(
                    prompts.CLAIM_MATCH_PROMPT.format(
                        reference_claims=prompts.numbered_list(answer_claims),
                        answer_claims=prompts.numbered_list(gt_claims),
                    )
                )
            ).get("matches", []) or []
            covered_in_answer = sum(1 for e in rev if e.get("matched"))

        precision = clamp(
            correct_in_answer / len(answer_claims) if answer_claims else 0.0
        )
        recall = clamp(covered_in_answer / len(gt_claims) if gt_claims else 0.0)
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return MetricResult(
            score=f1,
            reasoning=(
                f"Claim precision={precision:.3f}, recall={recall:.3f}, F1={f1:.3f}."
            ),
            violations=(
                ["Low claim precision: answer includes unverified facts."]
                if precision < 0.5
                else []
            )
            + (
                ["Low claim recall: answer omits reference facts."]
                if recall < 0.5
                else []
            ),
            metadata={
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "answer_claim_count": len(answer_claims),
                "gt_claim_count": len(gt_claims),
                "correct_in_answer": correct_in_answer,
                "covered_in_answer": covered_in_answer,
            },
            metric_name=self.name,
        )
