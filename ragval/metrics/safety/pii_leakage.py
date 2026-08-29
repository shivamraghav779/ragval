"""PII leakage: does the answer expose personal data it should not?"""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.utils import prompts
from ragval.utils.scoring import clamp


class PIILeakageMetric(BaseMetric):
    name = "pii_leakage"
    description = "Answer exposes personally identifiable information"
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
        if not answer or not answer.strip():
            return self._na("no answer to assess")

        data = await provider.complete_json(
            prompts.PII_LEAKAGE_PROMPT.format(question=question, answer=answer)
        )
        raw = clamp(float(data.get("leakage_score", 0.0) or 0.0))
        items = data.get("pii_items", []) or []
        detected = bool(data.get("pii_detected"))

        return MetricResult(
            score=1.0 - raw,
            reasoning=data.get("reasoning", "")
            or (f"PII leakage score {raw:.3f} (inverted to {1.0 - raw:.3f})."),
            violations=[
                f"{i.get('type', 'PII')} ({i.get('severity', '?')}): "
                f"{i.get('excerpt', '')}"
                for i in items
            ],
            metadata={"pii_detected": detected, "pii_items": items, "raw_score": raw},
            metric_name=self.name,
        )
