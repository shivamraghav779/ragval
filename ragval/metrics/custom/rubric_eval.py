"""RubricEval: score an answer by matching it to a 1-5 quality rubric."""

from __future__ import annotations

from typing import Any, Dict, List

from ragval.metrics.base import MetricResult
from ragval.utils import prompts
from ragval.utils.async_utils import run_sync


class RubricEval:
    category = "custom"

    def __init__(
        self,
        name: str,
        rubric: Dict[int, str],
        model: str,
        weight_top_heavy: bool = False,
    ) -> None:
        self.name = name
        self.rubric = {int(k): v for k, v in rubric.items()}
        self.model = model
        self.weight_top_heavy = weight_top_heavy
        self.requires_ground_truth = False

    def _rubric_text(self) -> str:
        return "\n".join(
            f"Level {lvl}: {desc}" for lvl, desc in sorted(self.rubric.items())
        )

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
            strictness = (
                "Be strict: reserve the top level for answers with no flaws at all."
                if self.weight_top_heavy
                else "Score fairly against each level description."
            )
            data = await prov.complete_json(
                prompts.RUBRIC_EVAL_PROMPT.format(
                    name=self.name,
                    strictness_note=strictness,
                    rubric=self._rubric_text(),
                    question=question,
                    answer=answer,
                    contexts=prompts.join_contexts(contexts or []),
                )
            )
            level = int(data.get("selected_level", 1) or 1)
            level = max(1, min(5, level))
            score = (level - 1) / 4.0

            return MetricResult(
                score=score,
                reasoning=data.get("reasoning", ""),
                violations=data.get("specific_issues", []) or [],
                metadata={
                    "selected_level": level,
                    "rubric_text": self._rubric_text(),
                    "reasoning": data.get("reasoning", ""),
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
