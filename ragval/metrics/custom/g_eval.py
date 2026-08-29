"""G-Eval: a user-defined metric driven by plain-English criteria.

Does NOT inherit BaseMetric — it is instantiated per criteria rather than
registered globally. It still returns a :class:`MetricResult`.
"""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.base import MetricResult
from ragval.utils import prompts
from ragval.utils.async_utils import run_sync
from ragval.utils.scoring import clamp


class GEval:
    category = "custom"

    def __init__(
        self,
        name: str,
        criteria: str,
        model: str,
        steps: Optional[List[str]] = None,
        threshold: float = 0.5,
    ) -> None:
        self.name = name
        self.criteria = criteria
        self.model = model
        self.steps = steps
        self.threshold = threshold
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

            steps = self.steps
            if not steps:
                steps_data = await prov.complete_json(
                    prompts.G_EVAL_STEPS_PROMPT.format(
                        name=self.name, criteria=self.criteria
                    )
                )
                steps = [s for s in steps_data.get("evaluation_steps", []) if s]

            score_data = await prov.complete_json(
                prompts.G_EVAL_SCORE_PROMPT.format(
                    name=self.name,
                    criteria=self.criteria,
                    steps=prompts.numbered_list(steps),
                    question=question,
                    answer=answer,
                    contexts=prompts.join_contexts(contexts or []),
                )
            )
            score = clamp(float(score_data.get("score", 0.0) or 0.0))
            violations = score_data.get("violations", []) or []

            return MetricResult(
                score=score,
                reasoning=score_data.get("reasoning", ""),
                violations=violations,
                metadata={
                    "criteria": self.criteria,
                    "evaluation_steps": steps,
                    "reasoning": score_data.get("reasoning", ""),
                    "threshold": self.threshold,
                    "passed": score >= self.threshold,
                },
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
