"""Conversation relevancy: do assistant turns stay on what the user asked?"""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.metrics.conversation.conversation_completeness import format_turns
from ragval.utils import prompts
from ragval.utils.scoring import clamp


class ConversationRelevancyMetric(BaseMetric):
    name = "conversation_relevancy"
    description = "Assistant turns stay relevant to the user's messages (no drift)"
    requires_ground_truth = False
    category = "conversation"

    async def _compute(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        provider: Any,
        ground_truth: Optional[str] = None,
        **kwargs: Any,
    ) -> MetricResult:
        turns = kwargs.get("turns", []) or []
        if not turns:
            return self._na("conversation_relevancy needs turns")
        assistant_turns = [t for t in turns if t.get("role") == "assistant"]
        if not assistant_turns:
            return self._na("no assistant turns to evaluate")

        data = await provider.complete_json(
            prompts.CONVERSATION_RELEVANCY_PROMPT.format(turns=format_turns(turns))
        )
        verdicts = data.get("turn_relevancy", []) or []
        overall = data.get("overall_relevancy")
        if overall is None and verdicts:
            overall = sum(1 for v in verdicts if v.get("relevant")) / len(verdicts)
        overall = clamp(float(overall or 0.0))

        return MetricResult(
            score=overall,
            reasoning=f"Overall turn relevancy {overall:.3f} across "
            f"{len(assistant_turns)} assistant turns.",
            violations=[
                f"turn {v.get('turn_index')}: {v.get('note') or 'off-topic'}"
                for v in verdicts
                if not v.get("relevant")
            ],
            metadata={"turn_relevancy": verdicts, "assistant_turn_count": len(assistant_turns)},
            metric_name=self.name,
        )
