"""Bias: demographic, political, or social bias introduced by the model."""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.utils import prompts
from ragval.utils.scoring import clamp


class BiasMetric(BaseMetric):
    name = "bias"
    description = "Demographic, political, or social bias in answer"
    requires_ground_truth = False
    category = "safety"

    async def _compute(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        provider: Any,
        ground_truth: Optional[str] = None,
        **kwargs: Any,
    ) -> MetricResult:
        data = await provider.complete_json(
            prompts.BIAS_EVALUATION_PROMPT.format(
                question=question,
                answer=answer,
                contexts=prompts.join_contexts(contexts),
            )
        )
        raw = clamp(float(data.get("overall_bias_score", 0.0) or 0.0))
        bias_types = data.get("bias_types", []) or []
        detected = bool(data.get("bias_detected"))

        return MetricResult(
            score=1.0 - raw,
            reasoning=data.get("reasoning", "")
            or f"Bias score {raw:.3f} (inverted to {1.0 - raw:.3f}).",
            violations=[
                f"{b.get('type', 'bias')} ({b.get('severity', '?')}): "
                f"{b.get('description', '')}"
                for b in bias_types
            ],
            metadata={
                "bias_detected": detected,
                "bias_types": bias_types,
                "raw_score": raw,
            },
            metric_name=self.name,
        )
