"""Context entity recall: are the named entities of the reference answer
present in the retrieved chunks?"""

from __future__ import annotations

from typing import Any, List, Optional

from ragval.metrics.base import BaseMetric, MetricResult
from ragval.utils import prompts, text


class ContextEntityRecallMetric(BaseMetric):
    name = "context_entity_recall"
    description = "Named entity coverage in retrieved context"
    requires_ground_truth = True
    category = "retrieval"

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
            return self._na("context_entity_recall requires ground_truth")

        seed = text.extract_entities(ground_truth)
        seed_text = prompts.numbered_list([str(e["text"]) for e in seed]) if seed else "(none)"
        extract = await provider.complete_json(
            prompts.ENTITY_EXTRACTION_PROMPT.format(
                text=ground_truth, seed_entities=seed_text
            )
        )
        entities = []
        for e in extract.get("entities", []) or []:
            t = (e.get("text") or "").strip()
            if t:
                entities.append({"text": t, "type": e.get("type", "unknown")})

        if not entities:
            return self._na("no named entities found in ground_truth")

        joined = "\n".join(contexts).lower()
        found = []
        missing = []
        for e in entities:
            if e["text"].lower() in joined:
                found.append(e["text"])
            else:
                missing.append(e["text"])

        score = len(found) / len(entities)
        return MetricResult(
            score=score,
            reasoning=(
                f"{len(found)}/{len(entities)} reference entities appear in the "
                f"retrieved context."
            ),
            violations=[f"entity missing from context: {m}" for m in missing],
            metadata={
                "entities": entities,
                "found_count": len(found),
                "total_count": len(entities),
                "missing_entities": missing,
            },
            metric_name=self.name,
        )
