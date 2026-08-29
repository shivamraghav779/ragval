"""Conversation metric behaviour."""

from __future__ import annotations

import pytest

from ragval.metrics.conversation import (
    ConversationCompletenessMetric,
    KnowledgeRetentionMetric,
    RoleAdherenceMetric,
)


@pytest.mark.asyncio
async def test_knowledge_retention_needs_four_turns(mock_provider):
    turns = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    result = await KnowledgeRetentionMetric().compute(
        "", "", [], mock_provider, turns=turns
    )
    assert result.score is None
    assert "insufficient turns" in result.reasoning


@pytest.mark.asyncio
async def test_knowledge_retention_retained(mock_provider, sample_turns):
    result = await KnowledgeRetentionMetric().compute(
        "", "", [], mock_provider, turns=sample_turns
    )
    assert result.score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_conversation_completeness(mock_provider, sample_turns):
    result = await ConversationCompletenessMetric().compute(
        "", "", [], mock_provider, turns=sample_turns
    )
    assert result.score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_role_adherence_requires_system_role(mock_provider, sample_turns):
    result = await RoleAdherenceMetric().compute(
        "", "", [], mock_provider, turns=sample_turns
    )
    assert result.score is None


@pytest.mark.asyncio
async def test_role_adherence_off_role_detected(mock_provider, sample_turns):
    mock_provider.set_response(
        "Evaluate each assistant turn below",
        {
            "adherence_verdicts": [
                {"turn_index": 1, "adheres": True, "violation_type": None},
                {"turn_index": 3, "adheres": False, "violation_type": "out_of_scope"},
                {"turn_index": 5, "adheres": True, "violation_type": None},
            ],
            "overall_adherence": 0.66,
        },
    )
    result = await RoleAdherenceMetric().compute(
        "", "", [], mock_provider, turns=sample_turns,
        system_role="clinical decision support assistant",
    )
    assert result.score == pytest.approx(0.66)
    assert any("out_of_scope" in v for v in result.violations)
