"""Base classes shared by every metric.

``MetricResult`` is the canonical result type (re-exported from
``ragval.result``). ``BaseMetric`` is the abstract contract: an async
``compute`` that must NEVER raise — any exception is caught and returned as
``MetricResult.error(...)`` so a single bad metric never aborts an evaluation.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

Category = str  # "retrieval" | "generation" | "safety" | "agentic" | "conversation" | "custom"


@dataclass
class MetricResult:
    """The outcome of computing one metric."""

    score: Optional[float]
    reasoning: str
    violations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    metric_name: str = ""
    requires_ground_truth: bool = False
    ground_truth_provided: bool = False
    computation_time_ms: float = 0.0

    @classmethod
    def not_applicable(cls, metric_name: str, reason: str) -> "MetricResult":
        return cls(
            score=None,
            reasoning=reason,
            violations=[],
            metadata={},
            metric_name=metric_name,
            requires_ground_truth=False,
            ground_truth_provided=False,
        )

    @classmethod
    def error(cls, metric_name: str, error_message: str) -> "MetricResult":
        return cls(
            score=None,
            reasoning=f"Error: {error_message}",
            violations=[],
            metadata={"error": error_message},
            metric_name=metric_name,
        )

    # -- serialization ----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "reasoning": self.reasoning,
            "violations": list(self.violations),
            "metadata": self.metadata,
            "metric_name": self.metric_name,
            "requires_ground_truth": self.requires_ground_truth,
            "ground_truth_provided": self.ground_truth_provided,
            "computation_time_ms": round(self.computation_time_ms, 2),
        }

    @property
    def is_applicable(self) -> bool:
        return self.score is not None

    @property
    def failed(self) -> bool:
        return self.metadata.get("error") is not None


class BaseMetric(abc.ABC):
    """Abstract metric. Subclasses set class attributes and implement ``_compute``."""

    name: str = "base"
    description: str = ""
    requires_ground_truth: bool = False
    category: Category = "generation"

    async def compute(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        provider: Any,
        ground_truth: Optional[str] = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Public entry point. Times the call and swallows every exception."""
        start = time.perf_counter()
        try:
            result = await self._compute(
                question=question,
                answer=answer,
                contexts=contexts or [],
                provider=provider,
                ground_truth=ground_truth,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - metrics must never propagate
            result = MetricResult.error(self.name, str(exc))
        result.metric_name = result.metric_name or self.name
        result.requires_ground_truth = self.requires_ground_truth
        result.ground_truth_provided = ground_truth is not None
        result.computation_time_ms = (time.perf_counter() - start) * 1000.0
        return result

    @abc.abstractmethod
    async def _compute(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        provider: Any,
        ground_truth: Optional[str] = None,
        **kwargs: Any,
    ) -> MetricResult:
        """The actual metric logic. May raise; ``compute`` will catch it."""

    def compute_sync(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        provider: Any,
        ground_truth: Optional[str] = None,
        **kwargs: Any,
    ) -> MetricResult:
        from ragval.utils.async_utils import run_sync

        return run_sync(
            self.compute(
                question, answer, contexts, provider, ground_truth, **kwargs
            )
        )

    # -- small helpers for subclasses -----------------------------------

    def _na(self, reason: str) -> MetricResult:
        return MetricResult.not_applicable(self.name, reason)
