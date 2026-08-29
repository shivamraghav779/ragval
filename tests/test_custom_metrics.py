"""Custom metric behaviour: GEval, RubricEval, AspectCritic."""

from __future__ import annotations

import pytest

from ragval.metrics.custom import AspectCritic, GEval, RubricEval


@pytest.mark.asyncio
async def test_geval_scores_from_criteria(mock_provider):
    metric = GEval(
        name="Dosing Specificity",
        criteria="The answer must include exact dose amounts with units.",
        model="mock/model",
    )
    result = await metric.compute(
        "q", "Metformin 500mg twice daily.", ["ctx"], mock_provider
    )
    assert result.score == pytest.approx(0.9)
    assert result.metadata["evaluation_steps"]


@pytest.mark.asyncio
async def test_geval_uses_provided_steps(mock_provider):
    metric = GEval(
        name="X", criteria="c", model="mock/model", steps=["step a", "step b"]
    )
    result = await metric.compute("q", "a", ["ctx"], mock_provider)
    assert result.metadata["evaluation_steps"] == ["step a", "step b"]


@pytest.mark.asyncio
async def test_rubric_eval_level_mapping(mock_provider):
    rubric = {1: "bad", 2: "poor", 3: "ok", 4: "good", 5: "excellent"}
    metric = RubricEval(name="Quality", rubric=rubric, model="mock/model")

    mock_provider.set_response(
        "selecting the single rubric level",
        {"selected_level": 1, "reasoning": "bad", "specific_issues": ["x"]},
    )
    low = await metric.compute("q", "a", ["ctx"], mock_provider)
    assert low.score == 0.0

    mock_provider.set_response(
        "selecting the single rubric level",
        {"selected_level": 5, "reasoning": "great", "specific_issues": []},
    )
    high = await metric.compute("q", "a", ["ctx"], mock_provider)
    assert high.score == 1.0


@pytest.mark.asyncio
async def test_aspect_critic_pass_fail(mock_provider):
    metric = AspectCritic(
        name="Has Citations", aspect="citations",
        description="answer cites sources", model="mock/model",
    )
    passed = await metric.compute("q", "a [1]", ["ctx"], mock_provider)
    assert passed.score == 1.0
    assert passed.violations == []

    mock_provider.set_response(
        "binary pass/fail verdict",
        {"passed": False, "score": 0.0, "reason": "no citations"},
    )
    failed = await metric.compute("q", "a", ["ctx"], mock_provider)
    assert failed.score == 0.0
    assert failed.violations == ["no citations"]


@pytest.mark.asyncio
async def test_custom_metric_error_is_captured():
    class Boom:
        async def complete_json(self, prompt):
            raise RuntimeError("kaboom")

    metric = GEval(name="X", criteria="c", model="mock/model", steps=["s"])
    result = await metric.compute("q", "a", ["ctx"], Boom())
    assert result.score is None
    assert "kaboom" in result.reasoning
