"""Domain profiles add domain-specific metrics and system-prompt guidance."""

from __future__ import annotations

import abc
from typing import Any, Dict, List

from ragval.metrics.base import MetricResult


class BaseDomain(abc.ABC):
    """A domain profile: extra metrics + evaluator prompt guidance."""

    name: str = "base"
    description: str = ""
    additional_metric_names: List[str] = []
    system_prompt_addition: str = ""

    @abc.abstractmethod
    async def get_domain_metrics(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        provider: Any,
    ) -> Dict[str, MetricResult]:
        """Compute this domain's extra metrics. Never raises — errors become
        ``MetricResult.error`` entries."""

    async def _safe(self, name: str, coro) -> MetricResult:
        try:
            return await coro
        except Exception as exc:  # noqa: BLE001
            return MetricResult.error(name, str(exc))
