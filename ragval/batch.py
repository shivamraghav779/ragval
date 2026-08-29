"""Batch evaluation orchestration.

``run_batch`` fans out single evaluations under a concurrency limit and folds
the results into a :class:`BatchEvaluationResult`. The evaluator delegates here.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List

from ragval.exceptions import InvalidInputError
from ragval.result import BatchEvaluationResult, EvaluationResult

__all__ = ["run_batch", "BatchEvaluationResult"]


def _normalize_pair(pair: Dict[str, Any], index: int) -> Dict[str, Any]:
    if not isinstance(pair, dict):
        raise InvalidInputError(
            f"qa_pairs[{index}]", "each item must be a dict"
        )
    if "question" not in pair or "answer" not in pair:
        raise InvalidInputError(
            f"qa_pairs[{index}]", "must contain 'question' and 'answer'"
        )
    contexts = pair.get("contexts") or pair.get("context") or []
    if isinstance(contexts, str):
        contexts = [contexts]
    return {
        "question": pair["question"],
        "answer": pair["answer"],
        "contexts": list(contexts),
        "ground_truth": pair.get("ground_truth"),
    }


async def run_batch(
    evaluator: Any,
    qa_pairs: List[Dict[str, Any]],
    concurrency: int = 3,
    show_progress: bool = True,
    **kwargs: Any,
) -> BatchEvaluationResult:
    """Evaluate every pair. A single failing item never aborts the batch."""
    pairs = [_normalize_pair(p, i) for i, p in enumerate(qa_pairs)]
    total = len(pairs)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    completed = 0
    start = time.perf_counter()
    lock = asyncio.Lock()

    async def one(idx: int, pair: Dict[str, Any]) -> EvaluationResult:
        nonlocal completed
        async with semaphore:
            try:
                result = await evaluator.evaluate(
                    question=pair["question"],
                    answer=pair["answer"],
                    contexts=pair["contexts"],
                    ground_truth=pair["ground_truth"],
                    **kwargs,
                )
            except Exception as exc:  # noqa: BLE001 - keep the batch alive
                result = evaluator._build_error_result(pair, str(exc))
        async with lock:
            completed += 1
            if show_progress:
                print(f"Evaluating {completed}/{total}...", flush=True)
        return result

    results = await asyncio.gather(
        *(one(i, p) for i, p in enumerate(pairs))
    )

    batch = BatchEvaluationResult(
        results=list(results),
        domain=evaluator.domain_name,
        model=evaluator.model,
    )
    batch.total_duration_ms = (time.perf_counter() - start) * 1000.0
    return batch
