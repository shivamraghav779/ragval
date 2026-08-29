"""Toxicity: harmful or offensive content, with professional-context awareness."""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.utils import prompts
from ragval.utils.scoring import clamp


class ToxicityMetric(BaseMetric):
    name = "toxicity"
    description = "Harmful, offensive, or inappropriate content"
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
            prompts.TOXICITY_EVALUATION_PROMPT.format(question=question, answer=answer)
        )
        raw = clamp(float(data.get("toxicity_score", 0.0) or 0.0))
        detected = bool(data.get("toxicity_detected"))
        context_appropriate = bool(data.get("context_appropriate"))
        types = data.get("toxicity_types", []) or []

        reasoning = data.get("reasoning", "") or f"Toxicity score {raw:.3f}."
        if detected and context_appropriate:
            reasoning += (
                " Note: flagged content may be professionally appropriate in "
                "this clinical/legal/educational context."
            )

        return MetricResult(
            score=1.0 - raw,
            reasoning=reasoning,
            violations=[
                f"{t.get('type', 'toxic')} ({t.get('severity', '?')}): "
                f"{t.get('excerpt', '')}"
                for t in types
            ],
            metadata={
                "toxicity_detected": detected,
                "toxicity_types": types,
                "context_appropriate": context_appropriate,
            },
            metric_name=self.name,
        )
