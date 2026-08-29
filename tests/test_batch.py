"""Batch evaluation behaviour."""

from __future__ import annotations

import pytest

from ragval.result import BatchEvaluationResult


def _rows():
    ctx = [
        "ADA guidelines recommend metformin 500mg twice daily as first-line.",
        "Metformin is preferred unless contraindicated.",
        "Unrelated cell biology text.",
    ]
    return [
        {"question": "First-line for T2DM?", "answer": "Metformin 500mg twice daily.",
         "contexts": ctx, "ground_truth": "Metformin, 500mg twice daily."},
        {"question": "First-line for T2DM?", "answer": "Insulin immediately.",
         "contexts": ctx},
        {"question": "First-line for T2DM?", "answer": "Metformin.", "contexts": ctx},
    ]


@pytest.mark.asyncio
async def test_batch_processes_all_items(make_evaluator, capsys):
    ev = make_evaluator(metrics=["faithfulness", "context_precision"])
    batch = await ev.batch_evaluate(_rows(), concurrency=2)
    assert isinstance(batch, BatchEvaluationResult)
    assert batch.total == 3
    assert batch.pass_count + batch.warn_count + batch.fail_count == 3


@pytest.mark.asyncio
async def test_single_failure_does_not_stop_batch(make_evaluator, mock_provider, monkeypatch):
    ev = make_evaluator(metrics=["faithfulness"])
    original = ev.evaluate
    calls = {"n": 0}

    async def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom on item 2")
        return await original(*args, **kwargs)

    monkeypatch.setattr(ev, "evaluate", flaky)
    batch = await ev.batch_evaluate(_rows(), concurrency=1)
    assert batch.total == 3


@pytest.mark.asyncio
async def test_diagnosis_summary_counts(make_evaluator):
    ev = make_evaluator(metrics=["faithfulness", "context_precision", "answer_relevance", "hallucination"])
    batch = await ev.batch_evaluate(_rows(), concurrency=3)
    assert "failed_layer_distribution" in batch.diagnosis_summary


@pytest.mark.asyncio
async def test_worst_cases_returns_lowest(make_evaluator, mock_provider):
    ev = make_evaluator(metrics=["faithfulness"])

    import re

    def handler(prompt):
        if "Break the following answer into a list of atomic claims" in prompt:
            if "Insulin" in prompt:
                return {"claims": ["insulin is first-line"]}
            return {"claims": ["metformin is first-line"]}
        if "checking whether each claim is supported" in prompt:
            items = re.findall(r"^\s*\d+\.\s+(.*)$", prompt, re.MULTILINE)
            return {"verdicts": [
                {"claim": c, "supported": "insulin" not in c.lower(), "reason": "x"}
                for c in items
            ]}
        return mock_provider._default(prompt)

    mock_provider.set_default(handler)
    batch = await ev.batch_evaluate(_rows(), concurrency=1)
    worst = batch.worst_cases(1)
    assert "Insulin" in worst[0].answer


@pytest.mark.asyncio
async def test_report_and_dataframe(make_evaluator):
    ev = make_evaluator(metrics=["faithfulness", "context_precision"])
    batch = await ev.batch_evaluate(_rows(), concurrency=3)
    assert "ragval batch report" in batch.report()
    df = batch.to_dataframe()
    assert len(df) == 3


@pytest.mark.asyncio
async def test_filter_by_verdict(make_evaluator):
    ev = make_evaluator(metrics=["faithfulness"])
    batch = await ev.batch_evaluate(_rows(), concurrency=3)
    passed = batch.filter_by_verdict("PASS")
    assert all(r.verdict == "PASS" for r in passed)
