"""Answer correctness: factual F1 vs the reference answer, plus semantic sim."""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.utils import prompts, text
from ragval.utils.scoring import clamp

_F1_WEIGHT = 0.75
_SEM_WEIGHT = 0.25


class AnswerCorrectnessMetric(BaseMetric):
    name = "answer_correctness"
    description = "Factual correctness vs reference answer"
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
            return self._na("answer_correctness requires ground_truth")

        answer_claims_data = await provider.complete_json(
            prompts.CLAIM_EXTRACT_PROMPT.format(text=answer)
        )
        gt_claims_data = await provider.complete_json(
            prompts.CLAIM_EXTRACT_PROMPT.format(text=ground_truth)
        )
        answer_claims = [c for c in answer_claims_data.get("claims", []) if c]
        gt_claims = [c for c in gt_claims_data.get("claims", []) if c]

        tp = fp = fn = 0
        matches = []
        if answer_claims:
            match_data = await provider.complete_json(
                prompts.CLAIM_MATCH_PROMPT.format(
                    reference_claims=prompts.numbered_list(gt_claims),
                    answer_claims=prompts.numbered_list(answer_claims),
                )
            )
            matches = match_data.get("matches", []) or []
            for i, _claim in enumerate(answer_claims):
                entry = matches[i] if i < len(matches) else {}
                if entry.get("matched"):
                    tp += 1
                else:
                    fp += 1

        # Recall side: which ground-truth claims are covered by the answer?
        if gt_claims:
            reverse_data = await provider.complete_json(
                prompts.CLAIM_MATCH_PROMPT.format(
                    reference_claims=prompts.numbered_list(answer_claims),
                    answer_claims=prompts.numbered_list(gt_claims),
                )
            )
            reverse = reverse_data.get("matches", []) or []
            covered = sum(1 for e in reverse if e.get("matched"))
            fn = max(0, len(gt_claims) - covered)

        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        semantic_sim = text.sentence_similarity(answer, ground_truth)
        score = clamp(_F1_WEIGHT * f1 + _SEM_WEIGHT * semantic_sim)

        return MetricResult(
            score=score,
            reasoning=(
                f"Factual F1={f1:.3f} (TP={tp}, FP={fp}, FN={fn}), "
                f"semantic similarity={semantic_sim:.3f}. "
                f"Weighted score={score:.3f}."
            ),
            violations=(
                ["Answer contains claims not in the reference."] if fp else []
            )
            + (["Answer omits facts from the reference."] if fn else []),
            metadata={
                "f1": f1,
                "semantic_similarity": semantic_sim,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "answer_claims": answer_claims,
                "ground_truth_claims": gt_claims,
            },
            metric_name=self.name,
        )
