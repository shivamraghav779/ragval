"""RAGEvaluator behaviour (with the MockProvider swapped in)."""

from __future__ import annotations

import pytest

from ragval.result import EvaluationResult


@pytest.mark.asyncio
async def test_evaluate_returns_evaluation_result(make_evaluator, sample_question, sample_answer, sample_contexts):
    ev = make_evaluator()
    result = await ev.evaluate(sample_question, sample_answer, sample_contexts)
    assert isinstance(result, EvaluationResult)
    assert result.faithfulness is not None
    assert result.diagnosis is not None


@pytest.mark.asyncio
async def test_verdict_pass_for_high_scores(make_evaluator, sample_question, sample_answer, sample_contexts):
    ev = make_evaluator()
    result = await ev.evaluate(sample_question, sample_answer, sample_contexts)
    assert result.verdict == "PASS"


@pytest.mark.asyncio
async def test_verdict_fail_for_hallucination(make_evaluator, mock_provider, sample_question, sample_answer, sample_contexts):
    mock_provider.set_response(
        "For each factual detail from an answer, check it against the context",
        {"checks": [
            {"entity": "500mg", "found": False, "contradicted": True,
             "contradiction_detail": "context says 850mg"}
        ]},
    )
    ev = make_evaluator()
    result = await ev.evaluate(sample_question, sample_answer, sample_contexts)
    assert result.hallucination_detected is True
    assert result.verdict == "FAIL"


@pytest.mark.asyncio
async def test_domain_metrics_for_clinical(make_evaluator, sample_question, sample_answer, sample_contexts):
    ev = make_evaluator(domain="clinical")
    result = await ev.evaluate(sample_question, sample_answer, sample_contexts)
    assert "drug_name_precision" in result.domain_metrics
    assert result.domain_metrics["drug_name_precision"].score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_metrics_subset_runs_only_requested(make_evaluator, sample_question, sample_answer, sample_contexts):
    ev = make_evaluator(metrics=["faithfulness"])
    result = await ev.evaluate(sample_question, sample_answer, sample_contexts)
    assert result.faithfulness is not None
    assert result.context_precision is None
    assert result.answer_relevance is None


@pytest.mark.asyncio
async def test_unknown_metric_rejected(make_evaluator):
    with pytest.raises(Exception):
        make_evaluator(metrics=["not_a_real_metric"])


def test_evaluate_sync_without_event_loop(make_evaluator, sample_question, sample_answer, sample_contexts):
    ev = make_evaluator(metrics=["faithfulness"])
    result = ev.evaluate_sync(sample_question, sample_answer, sample_contexts)
    assert result.faithfulness is not None


@pytest.mark.asyncio
async def test_evaluate_agent(make_evaluator, sample_question, sample_answer, sample_tool_calls):
    ev = make_evaluator(metrics=["faithfulness"])
    result = await ev.evaluate_agent(
        question=sample_question,
        answer=sample_answer,
        tool_calls=sample_tool_calls,
        expected_tools=["search_guidelines", "lookup_drug"],
        action_trace=["search", "lookup", "answer"],
        declared_plan="1. search 2. lookup 3. answer",
        contexts=["ADA guidelines recommend metformin."],
    )
    assert result.tool_correctness.score == pytest.approx(1.0)
    assert result.task_completion is not None


@pytest.mark.asyncio
async def test_evaluate_conversation(make_evaluator, sample_turns):
    ev = make_evaluator()
    result = await ev.evaluate_conversation(
        turns=sample_turns, system_role="clinical decision support assistant"
    )
    assert result.turn_count == 6
    assert result.role_adherence.score is not None
    assert result.verdict in {"PASS", "WARN", "FAIL"}


@pytest.mark.asyncio
async def test_evaluate_pipeline_with_expected_chunks(make_evaluator, sample_question, sample_answer, sample_ground_truth):
    ev = make_evaluator(metrics=["faithfulness"])
    chunks = ["ADA guidelines recommend metformin 500mg twice daily.",
              "Metformin is preferred unless contraindicated.",
              "Unrelated text about cell biology."]
    result = await ev.evaluate_pipeline(
        question=sample_question,
        answer=sample_answer,
        retrieved_chunks=chunks,
        ground_truth=sample_ground_truth,
        expected_chunks=chunks[:2],
        k=2,
    )
    assert result.mrr is not None
    assert result.ndcg is not None
    assert result.hit_rate is not None


@pytest.mark.asyncio
async def test_module_level_evaluate(monkeypatch, sample_question, sample_answer, sample_contexts):
    import ragval.evaluator as ev_mod
    from tests.conftest import MockProvider

    real_init = ev_mod.RAGEvaluator.__init__

    def patched_init(self, *a, **kw):
        real_init(self, *a, **kw)
        self.provider = MockProvider()

    monkeypatch.setattr(ev_mod.RAGEvaluator, "__init__", patched_init)
    result = await ev_mod.evaluate(
        sample_question, sample_answer, sample_contexts, model="mock/model"
    )
    assert result.verdict in {"PASS", "WARN", "FAIL"}
