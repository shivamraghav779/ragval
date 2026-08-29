"""Safety metric behaviour."""

from __future__ import annotations

import pytest

from ragval.metrics.safety import BiasMetric, TopicAdherenceMetric, ToxicityMetric


@pytest.mark.asyncio
async def test_bias_detected(mock_provider, sample_question, sample_contexts):
    mock_provider.set_response(
        "Evaluate the answer for bias",
        {
            "bias_detected": True,
            "bias_types": [
                {"type": "demographic", "description": "gender stereotyping",
                 "severity": "high"}
            ],
            "overall_bias_score": 0.8,
            "reasoning": "stereotypes present",
        },
    )
    result = await BiasMetric().compute(
        sample_question, "Men are better at this.", sample_contexts, mock_provider
    )
    assert result.score == pytest.approx(0.2)
    assert result.violations


@pytest.mark.asyncio
async def test_bias_clean_answer(mock_provider, sample_question, sample_answer, sample_contexts):
    result = await BiasMetric().compute(
        sample_question, sample_answer, sample_contexts, mock_provider
    )
    assert result.score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_toxicity_clinical_mention_not_flagged(mock_provider, sample_question, sample_contexts):
    result = await ToxicityMetric().compute(
        sample_question,
        "Metformin overdose can cause lactic acidosis; monitor closely.",
        sample_contexts,
        mock_provider,
    )
    assert result.score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_toxicity_appropriate_context_note(mock_provider, sample_question, sample_contexts):
    mock_provider.set_response(
        "Evaluate the answer for toxic content",
        {
            "toxicity_detected": True,
            "toxicity_types": [{"type": "medical", "excerpt": "overdose", "severity": "low"}],
            "toxicity_score": 0.1,
            "context_appropriate": True,
            "reasoning": "clinical",
        },
    )
    result = await ToxicityMetric().compute(
        sample_question, "overdose management", sample_contexts, mock_provider
    )
    assert "professionally appropriate" in result.reasoning


@pytest.mark.asyncio
async def test_topic_adherence_off_topic_fails(mock_provider, sample_question, sample_contexts):
    mock_provider.set_response(
        "stays within the expected domain",
        {
            "on_topic": False,
            "off_topic_content": ["stock tips"],
            "adherence_score": 0.2,
            "reasoning": "drifts to finance",
        },
    )
    result = await TopicAdherenceMetric().compute(
        sample_question, "Also, buy tech stocks.", sample_contexts, mock_provider,
        domain="clinical",
    )
    assert result.score == pytest.approx(0.2)
    assert "off-topic: stock tips" in result.violations


@pytest.mark.asyncio
async def test_topic_adherence_needs_scope(mock_provider, sample_question, sample_answer, sample_contexts):
    result = await TopicAdherenceMetric().compute(
        sample_question, sample_answer, sample_contexts, mock_provider, domain="general"
    )
    assert result.score is None
