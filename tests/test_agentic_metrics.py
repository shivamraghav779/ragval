"""Agentic metric behaviour."""

from __future__ import annotations

import pytest

from ragval.metrics.agentic import (
    AgentGoalAccuracyMetric,
    ArgumentCorrectnessMetric,
    PlanAdherenceMetric,
    PlanQualityMetric,
    StepEfficiencyMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
)


@pytest.mark.asyncio
async def test_tool_correctness_exact_match(mock_provider):
    result = await ToolCorrectnessMetric().compute(
        "q", "a", [], mock_provider,
        tool_calls=[{"name": "search"}, {"name": "lookup"}],
        expected_tools=["search", "lookup"],
    )
    assert result.score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_tool_correctness_wrong_tools(mock_provider):
    result = await ToolCorrectnessMetric().compute(
        "q", "a", [], mock_provider,
        tool_calls=[{"name": "delete"}],
        expected_tools=["search", "lookup"],
    )
    assert result.score < 0.5
    assert result.metadata["FP"] == 1
    assert result.metadata["FN"] == 2


@pytest.mark.asyncio
async def test_tool_correctness_missing_kwargs(mock_provider):
    result = await ToolCorrectnessMetric().compute("q", "a", [], mock_provider)
    assert result.score is None


@pytest.mark.asyncio
async def test_argument_correctness(mock_provider):
    result = await ArgumentCorrectnessMetric().compute(
        "q", "a", [], mock_provider,
        tool_calls=[{"name": "search", "arguments": {"q": "x"}}],
        expected_tool_calls=[{"name": "search", "arguments": {"q": "x"}}],
    )
    assert result.score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_argument_correctness_no_overlap(mock_provider):
    result = await ArgumentCorrectnessMetric().compute(
        "q", "a", [], mock_provider,
        tool_calls=[{"name": "a", "arguments": {}}],
        expected_tool_calls=[{"name": "b", "arguments": {}}],
    )
    assert result.score is None


@pytest.mark.asyncio
async def test_task_completion(mock_provider):
    result = await TaskCompletionMetric().compute(
        "book a flight", "flight booked", [], mock_provider,
        action_trace=["search flights", "select flight", "pay"],
    )
    assert result.score > 0.9


@pytest.mark.asyncio
async def test_step_efficiency_minimal_steps(mock_provider):
    result = await StepEfficiencyMetric().compute(
        "q", "a", [], mock_provider,
        action_trace=["step1", "step2"], expected_min_steps=2,
    )
    assert result.score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_step_efficiency_wasteful(mock_provider):
    result = await StepEfficiencyMetric().compute(
        "q", "a", [], mock_provider,
        action_trace=["s1", "s2", "s3", "s4", "s5", "s6"], expected_min_steps=2,
    )
    assert result.score == pytest.approx(2 / 6)


@pytest.mark.asyncio
async def test_step_efficiency_missing_trace(mock_provider):
    result = await StepEfficiencyMetric().compute("q", "a", [], mock_provider)
    assert result.score is None


@pytest.mark.asyncio
async def test_plan_adherence_missing_plan(mock_provider):
    result = await PlanAdherenceMetric().compute(
        "q", "a", [], mock_provider, action_trace=["s1"]
    )
    assert result.score is None


@pytest.mark.asyncio
async def test_plan_quality(mock_provider):
    result = await PlanQualityMetric().compute(
        "q", "a", [], mock_provider, declared_plan="1. search 2. answer"
    )
    assert result.score > 0.8


@pytest.mark.asyncio
async def test_agent_goal_accuracy_requires_ground_truth(mock_provider):
    result = await AgentGoalAccuracyMetric().compute("q", "a", [], mock_provider)
    assert result.score is None
