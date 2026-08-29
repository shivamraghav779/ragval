"""Topic adherence: does the answer stay within the expected domain?"""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.utils import prompts
from ragval.utils.scoring import clamp

_DOMAIN_TOPICS = {
    "clinical": [
        "medicine", "health", "drugs", "clinical", "treatment", "diagnosis",
    ],
    "legal": [
        "law", "legal", "regulation", "statute", "court", "contract",
    ],
    "financial": [
        "finance", "investment", "banking", "accounting", "tax", "regulatory",
    ],
}


class TopicAdherenceMetric(BaseMetric):
    name = "topic_adherence"
    description = "Answer stays within expected domain"
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
        allowed_topics = list(kwargs.get("allowed_topics", []) or [])
        domain = kwargs.get("domain", "general")

        if not allowed_topics and domain in _DOMAIN_TOPICS:
            allowed_topics = _DOMAIN_TOPICS[domain]

        if not allowed_topics:
            return self._na(
                "topic_adherence needs allowed_topics or a known domain"
            )

        data = await provider.complete_json(
            prompts.TOPIC_ADHERENCE_PROMPT.format(
                domain=domain,
                allowed_topics=", ".join(allowed_topics),
                answer=answer,
            )
        )
        score = clamp(float(data.get("adherence_score", 1.0) or 0.0))
        off_topic = data.get("off_topic_content", []) or []

        return MetricResult(
            score=score,
            reasoning=data.get("reasoning", "") or f"Topic adherence {score:.3f}.",
            violations=[f"off-topic: {o}" for o in off_topic],
            metadata={
                "on_topic": bool(data.get("on_topic")),
                "off_topic_excerpts": off_topic,
                "domain": domain,
            },
            metric_name=self.name,
        )
