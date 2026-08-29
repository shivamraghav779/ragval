"""The default domain: no extra metrics, no prompt additions."""

from __future__ import annotations

from typing import Any, Dict, List

from ragval.domains.base import BaseDomain
from ragval.metrics.base import MetricResult


class GeneralDomain(BaseDomain):
    name = "general"
    description = "General-purpose RAG, no domain specialization"
    additional_metric_names: List[str] = []
    system_prompt_addition = ""

    async def get_domain_metrics(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        provider: Any,
    ) -> Dict[str, MetricResult]:
        return {}
