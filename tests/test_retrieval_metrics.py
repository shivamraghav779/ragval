"""Retrieval metric behaviour."""

from __future__ import annotations

import pytest

from ragval.metrics.retrieval import (
    ContextEntityRecallMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    ContextRelevanceMetric,
    HitRateMetric,
    MRRMetric,
    NDCGMetric,
    NoiseSensitivityMetric,
)


@pytest.mark.asyncio
async def test_context_precision_all_relevant(mock_provider, sample_question, sample_answer, sample_contexts):
    result = await ContextPrecisionMetric().compute(
        sample_question, sample_answer, sample_contexts, mock_provider
    )
    assert result.score == pytest.approx(1.0)
    assert result.metadata["relevant_count"] == 3


@pytest.mark.asyncio
async def test_context_precision_all_irrelevant(mock_provider, sample_question, sample_answer, sample_contexts):
    mock_provider.set_response(
        "evaluating a retrieval system",
        {"verdicts": [{"index": i, "relevant": False, "reason": "no"} for i in range(3)]},
    )
    result = await ContextPrecisionMetric().compute(
        sample_question, sample_answer, sample_contexts, mock_provider
    )
    assert result.score == 0.0


@pytest.mark.asyncio
async def test_context_precision_empty_contexts(mock_provider, sample_question, sample_answer):
    result = await ContextPrecisionMetric().compute(
        sample_question, sample_answer, [], mock_provider
    )
    assert result.score is None


@pytest.mark.asyncio
async def test_context_recall_requires_ground_truth(mock_provider, sample_question, sample_answer, sample_contexts):
    result = await ContextRecallMetric().compute(
        sample_question, sample_answer, sample_contexts, mock_provider, ground_truth=None
    )
    assert result.score is None
    assert result.requires_ground_truth is True


@pytest.mark.asyncio
async def test_context_recall_with_ground_truth(mock_provider, sample_question, sample_answer, sample_contexts, sample_ground_truth):
    result = await ContextRecallMetric().compute(
        sample_question, sample_answer, sample_contexts, mock_provider,
        ground_truth=sample_ground_truth,
    )
    assert result.score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_context_relevance_scores_high(mock_provider, sample_question, sample_answer, sample_contexts):
    result = await ContextRelevanceMetric().compute(
        sample_question, sample_answer, sample_contexts, mock_provider
    )
    assert result.score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_context_entity_recall_missing_entity(mock_provider, sample_question, sample_answer, sample_contexts):
    mock_provider.set_response(
        "Extract every named entity",
        {"entities": [{"text": "Ozempic", "type": "drug"}]},
    )
    result = await ContextEntityRecallMetric().compute(
        sample_question, sample_answer, sample_contexts, mock_provider,
        ground_truth="Ozempic is a GLP-1 agonist.",
    )
    assert result.score == 0.0
    assert "Ozempic" in result.metadata["missing_entities"]


@pytest.mark.asyncio
async def test_mrr_first_chunk_relevant(mock_provider, sample_question, sample_answer, sample_contexts):
    result = await MRRMetric().compute(
        sample_question, sample_answer, sample_contexts, mock_provider,
        chunk_relevance_labels=[True, False, False],
    )
    assert result.score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_mrr_second_chunk_relevant(mock_provider, sample_question, sample_answer, sample_contexts):
    result = await MRRMetric().compute(
        sample_question, sample_answer, sample_contexts, mock_provider,
        chunk_relevance_labels=[False, True, False],
    )
    assert result.score == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_ndcg_perfect_order(mock_provider, sample_question, sample_answer, sample_contexts):
    result = await NDCGMetric().compute(
        sample_question, sample_answer, sample_contexts, mock_provider,
        graded_relevances=[3, 2, 0],
    )
    assert result.score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_hit_rate_hit_and_miss(mock_provider, sample_question, sample_answer, sample_contexts):
    hit = await HitRateMetric().compute(
        sample_question, sample_answer, sample_contexts, mock_provider,
        k=2, chunk_relevance_labels=[False, True, False],
    )
    miss = await HitRateMetric().compute(
        sample_question, sample_answer, sample_contexts, mock_provider,
        k=1, chunk_relevance_labels=[False, True, False],
    )
    assert hit.score == 1.0
    assert miss.score == 0.0


@pytest.mark.asyncio
async def test_noise_sensitivity_insufficient_chunks(mock_provider, sample_question, sample_answer):
    result = await NoiseSensitivityMetric().compute(
        sample_question, sample_answer, ["only one"], mock_provider, top_k=3
    )
    assert result.score is None


@pytest.mark.asyncio
async def test_noise_sensitivity_robust(mock_provider, sample_question, sample_answer, sample_contexts):
    result = await NoiseSensitivityMetric().compute(
        sample_question, sample_answer, sample_contexts + ["extra noise chunk"],
        mock_provider, top_k=2,
    )
    assert result.score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_provider_error_returns_error_result(sample_question, sample_answer, sample_contexts):
    class Boom:
        async def complete_json(self, prompt):
            raise RuntimeError("provider down")

    result = await ContextPrecisionMetric().compute(
        sample_question, sample_answer, sample_contexts, Boom()
    )
    assert result.score is None
    assert "provider down" in result.reasoning
