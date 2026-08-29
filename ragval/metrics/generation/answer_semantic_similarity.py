"""Answer semantic similarity: TF-IDF cosine between answer and ground truth.

The fastest metric in the library — no LLM call. High similarity does NOT
imply factual accuracy; pair it with factual_correctness.
"""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.utils import text


class AnswerSemanticSimilarityMetric(BaseMetric):
    name = "answer_semantic_similarity"
    description = "TF-IDF cosine between answer and ground truth"
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
            return self._na("answer_semantic_similarity requires ground_truth")

        similarity = text.sentence_similarity(answer, ground_truth)
        return MetricResult(
            score=similarity,
            reasoning=(
                f"TF-IDF cosine similarity to the reference is {similarity:.3f}. "
                "Note: semantic similarity does not guarantee factual accuracy; "
                "a confidently wrong answer can still score high. Combine with "
                "factual_correctness."
            ),
            violations=[],
            metadata={"similarity_score": similarity},
            metric_name=self.name,
        )
