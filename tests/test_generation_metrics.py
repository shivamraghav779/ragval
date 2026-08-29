"""Generation metric behaviour."""

from __future__ import annotations

import pytest

from ragval.metrics.generation import (
    AnswerCorrectnessMetric,
    AnswerRelevanceMetric,
    AnswerSemanticSimilarityMetric,
    FactualCorrectnessMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    SummarizationMetric,
)


@pytest.mark.asyncio
async def test_faithfulness_grounded_answer(mock_provider, sample_question, sample_answer, sample_contexts):
    result = await FaithfulnessMetric().compute(
        sample_question, sample_answer, sample_contexts, mock_provider
    )
    assert result.score == pytest.approx(1.0)
    assert result.metadata["total_claims"] == 2


@pytest.mark.asyncio
async def test_faithfulness_hallucinated_answer(mock_provider, sample_question, sample_answer, sample_contexts):
    mock_provider.set_response(
        "checking whether each claim is supported",
        {"verdicts": [
            {"claim": "claim one", "supported": False, "reason": "not in context"},
            {"claim": "claim two", "supported": False, "reason": "not in context"},
        ]},
    )
    result = await FaithfulnessMetric().compute(
        sample_question, sample_answer, sample_contexts, mock_provider
    )
    assert result.score == 0.0
    assert len(result.violations) == 2


@pytest.mark.asyncio
async def test_faithfulness_empty_answer_is_vacuously_true(mock_provider, sample_question, sample_contexts):
    result = await FaithfulnessMetric().compute(
        sample_question, "", sample_contexts, mock_provider
    )
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_hallucination_contradiction_detected(mock_provider, sample_question, sample_answer, sample_contexts):
    mock_provider.set_response(
        "For each factual detail from an answer, check it against the context",
        {"checks": [
            {"entity": "500mg", "found": True, "contradicted": True,
             "contradiction_detail": "context says 850mg"},
        ]},
    )
    result = await HallucinationMetric().compute(
        sample_question, sample_answer, sample_contexts, mock_provider
    )
    assert result.metadata["hallucination_detected"] is True
    assert result.score < 1.0


@pytest.mark.asyncio
async def test_hallucination_no_specifics(mock_provider, sample_question, sample_contexts):
    mock_provider.set_response(
        "Extract every specific, checkable factual detail", {"entities": []}
    )
    result = await HallucinationMetric().compute(
        sample_question, "It depends on the situation.", sample_contexts, mock_provider
    )
    assert result.score == 0.0
    assert result.metadata["hallucination_detected"] is False


@pytest.mark.asyncio
async def test_answer_relevance_on_topic(mock_provider, sample_question, sample_answer, sample_contexts):
    mock_provider.set_response(
        "Generate 3 different questions",
        {"questions": [sample_question, sample_question, sample_question]},
    )
    result = await AnswerRelevanceMetric().compute(
        sample_question, sample_answer, sample_contexts, mock_provider
    )
    assert result.score > 0.9


@pytest.mark.asyncio
async def test_answer_correctness_requires_ground_truth(mock_provider, sample_question, sample_answer, sample_contexts):
    result = await AnswerCorrectnessMetric().compute(
        sample_question, sample_answer, sample_contexts, mock_provider, ground_truth=None
    )
    assert result.score is None


@pytest.mark.asyncio
async def test_answer_correctness_with_ground_truth(mock_provider, sample_question, sample_answer, sample_contexts, sample_ground_truth):
    result = await AnswerCorrectnessMetric().compute(
        sample_question, sample_answer, sample_contexts, mock_provider,
        ground_truth=sample_ground_truth,
    )
    assert result.score is not None and result.score > 0.5


@pytest.mark.asyncio
async def test_factual_correctness_reports_precision_recall(mock_provider, sample_question, sample_answer, sample_contexts, sample_ground_truth):
    result = await FactualCorrectnessMetric().compute(
        sample_question, sample_answer, sample_contexts, mock_provider,
        ground_truth=sample_ground_truth,
    )
    assert "precision" in result.metadata
    assert "recall" in result.metadata


@pytest.mark.asyncio
async def test_semantic_similarity_none_gt(mock_provider, sample_question, sample_answer, sample_contexts):
    result = await AnswerSemanticSimilarityMetric().compute(
        sample_question, sample_answer, sample_contexts, mock_provider
    )
    assert result.score is None


@pytest.mark.asyncio
async def test_semantic_similarity_identical(mock_provider, sample_question, sample_contexts):
    text = "Metformin is first-line for type 2 diabetes."
    result = await AnswerSemanticSimilarityMetric().compute(
        sample_question, text, sample_contexts, mock_provider, ground_truth=text
    )
    assert result.score == pytest.approx(1.0, abs=1e-6)


@pytest.mark.asyncio
async def test_summarization_combines_faithfulness_and_coverage(mock_provider, sample_question, sample_answer, sample_contexts):
    result = await SummarizationMetric().compute(
        sample_question, sample_answer, sample_contexts, mock_provider
    )
    assert result.score is not None
    assert "faithfulness_score" in result.metadata
    assert "coverage_score" in result.metadata
