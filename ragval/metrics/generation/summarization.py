"""Summarization: faithfulness + key-topic coverage, for long-doc RAG."""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.utils import prompts

_FAITH_WEIGHT = 0.6
_COVERAGE_WEIGHT = 0.4


class SummarizationMetric(BaseMetric):
    name = "summarization"
    description = "For long-doc RAG: captures key info without hallucinating"
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
        if not contexts:
            return self._na("summarization needs source context")

        from ragval.metrics.generation.faithfulness import FaithfulnessMetric

        faith_result = await FaithfulnessMetric().compute(
            question, answer, contexts, provider
        )
        faithfulness_score = (
            faith_result.score if faith_result.score is not None else 0.0
        )

        topics_data = await provider.complete_json(
            prompts.SUMMARIZATION_KEY_TOPICS_PROMPT.format(
                contexts=prompts.join_contexts(contexts)
            )
        )
        key_topics = [t for t in topics_data.get("key_topics", []) if t]

        if not key_topics:
            coverage_score = 1.0
            addressed: List[str] = []
            missed: List[str] = []
        else:
            cov_data = await provider.complete_json(
                prompts.SUMMARIZATION_COVERAGE_PROMPT.format(
                    answer=answer,
                    key_topics=prompts.numbered_list(key_topics),
                )
            )
            coverage = cov_data.get("coverage", []) or []
            addressed = [
                c.get("topic", key_topics[i] if i < len(key_topics) else "")
                for i, c in enumerate(coverage)
                if c.get("addressed")
            ]
            missed = [t for t in key_topics if t not in addressed]
            coverage_score = len(addressed) / len(key_topics)

        score = _FAITH_WEIGHT * faithfulness_score + _COVERAGE_WEIGHT * coverage_score
        return MetricResult(
            score=score,
            reasoning=(
                f"Faithfulness={faithfulness_score:.3f} (w={_FAITH_WEIGHT}), "
                f"coverage={coverage_score:.3f} (w={_COVERAGE_WEIGHT}). "
                f"Score={score:.3f}."
            ),
            violations=[f"summary omits topic: {t}" for t in missed],
            metadata={
                "faithfulness_score": faithfulness_score,
                "coverage_score": coverage_score,
                "key_topics": key_topics,
                "topics_addressed": addressed,
                "topics_missed": missed,
            },
            metric_name=self.name,
        )
