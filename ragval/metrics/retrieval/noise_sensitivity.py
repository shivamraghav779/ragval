"""Noise sensitivity: how much does answer faithfulness degrade when
irrelevant chunks are mixed into the context?"""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult


class NoiseSensitivityMetric(BaseMetric):
    name = "noise_sensitivity"
    description = "Answer quality robustness to irrelevant chunks"
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
        top_k = int(kwargs.get("top_k", 3))
        if len(contexts) <= top_k:
            return self._na("insufficient chunks for noise test")

        from ragval.metrics.generation.faithfulness import FaithfulnessMetric

        faith = FaithfulnessMetric()

        clean_contexts = contexts[:top_k]
        noisy_contexts = contexts[:top_k] + contexts[top_k:]

        # A precomputed clean faithfulness score can be passed in to save a call.
        clean_score = kwargs.get("clean_faithfulness_score")
        if clean_score is None:
            clean_result = await faith.compute(
                question, answer, clean_contexts, provider
            )
            clean_score = clean_result.score if clean_result.score is not None else 1.0

        noisy_result = await faith.compute(question, answer, noisy_contexts, provider)
        noisy_score = noisy_result.score if noisy_result.score is not None else clean_score

        sensitivity = abs(clean_score - noisy_score)
        score = max(0.0, 1.0 - sensitivity)

        return MetricResult(
            score=score,
            reasoning=(
                f"Faithfulness went from {clean_score:.3f} (clean) to "
                f"{noisy_score:.3f} with {len(contexts) - top_k} noise chunks "
                f"added. Robustness = {score:.3f}."
            ),
            violations=(
                [f"Faithfulness dropped {sensitivity:.3f} when noise was added."]
                if sensitivity > 0.2
                else []
            ),
            metadata={
                "clean_faithfulness": clean_score,
                "noisy_faithfulness": noisy_score,
                "sensitivity_delta": sensitivity,
                "top_k_used": top_k,
                "noise_chunks_added": len(contexts) - top_k,
            },
            metric_name=self.name,
        )
