"""AspectCritic: a binary pass/fail verdict on one specific aspect."""

from __future__ import annotations

from typing import Any, List

from ragval.metrics.base import MetricResult
from ragval.utils import prompts
from ragval.utils.async_utils import run_sync
from ragval.utils.scoring import clamp


class AspectCritic:
    category = "custom"

    def __init__(self, name: str, aspect: str, description: str, model: str) -> None:
        self.name = name
        self.aspect = aspect
        self.description = description
        self.model = model
        self.requires_ground_truth = False

    def _get_provider(self, provider: Any) -> Any:
        if provider is not None:
            return provider
        from ragval.providers import get_provider

        return get_provider(self.model)

    async def compute(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        provider: Any = None,
        **kwargs: Any,
    ) -> MetricResult:
        try:
            prov = self._get_provider(provider)
            data = await prov.complete_json(
                prompts.ASPECT_CRITIC_PROMPT.format(
                    aspect=self.aspect,
                    description=self.description,
                    question=question,
                    answer=answer,
                    contexts=prompts.join_contexts(contexts or []),
                )
            )
            passed = bool(data.get("passed"))
            score = clamp(float(data.get("score", 1.0 if passed else 0.0) or 0.0))
            reason = data.get("reason", "")

            return MetricResult(
                score=score,
                reasoning=reason,
                violations=[] if passed else [reason or f"failed aspect: {self.aspect}"],
                metadata={"aspect": self.aspect, "passed": passed, "reason": reason},
                metric_name=self.name,
            )
        except Exception as exc:  # noqa: BLE001
            return MetricResult.error(self.name, str(exc))

    def compute_sync(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        provider: Any = None,
        **kwargs: Any,
    ) -> MetricResult:
        return run_sync(
            self.compute(question, answer, contexts, provider, **kwargs)
        )
