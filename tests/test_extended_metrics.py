"""Extended metrics: context_utilization ... conversation_relevancy."""

from __future__ import annotations

import pytest

from ragval.metrics.conversation import ConversationRelevancyMetric
from ragval.metrics.generation import (
    AnswerCompletenessMetric,
    CitationSupportMetric,
    CoherenceMetric,
    ConcisenessMetric,
    FluencyMetric,
    RefusalAppropriatenessMetric,
)
from ragval.metrics.retrieval import (
    ContextSufficiencyMetric,
    ContextUtilizationMetric,
    RetrievalDiversityMetric,
)
from ragval.metrics.safety import PIILeakageMetric, ToneProfessionalismMetric


@pytest.mark.asyncio
async def test_context_utilization(mock_provider, sample_question, sample_answer, sample_contexts):
    r = await ContextUtilizationMetric().compute(
        sample_question, sample_answer, sample_contexts, mock_provider
    )
    assert r.score is not None and 0.0 <= r.score <= 1.0
    assert r.metadata["used_count"] >= 1


@pytest.mark.asyncio
async def test_context_utilization_empty_answer(mock_provider, sample_question, sample_contexts):
    r = await ContextUtilizationMetric().compute(
        sample_question, "", sample_contexts, mock_provider
    )
    assert r.score is None


@pytest.mark.asyncio
async def test_retrieval_diversity_distinct_vs_duplicate(mock_provider):
    distinct = await RetrievalDiversityMetric().compute(
        "q", "a",
        ["Metformin is first-line for diabetes.",
         "Hypertension is treated with ACE inhibitors.",
         "Depression responds to CBT and SSRIs."],
        mock_provider,
    )
    dup = await RetrievalDiversityMetric().compute(
        "q", "a",
        ["Metformin is first-line for type 2 diabetes.",
         "Metformin is the first-line drug for type 2 diabetes.",
         "Metformin is first line in type 2 diabetes management."],
        mock_provider,
    )
    assert distinct.score > dup.score
    assert dup.metadata["redundant_pairs"]


@pytest.mark.asyncio
async def test_retrieval_diversity_needs_two_chunks(mock_provider):
    r = await RetrievalDiversityMetric().compute("q", "a", ["one chunk"], mock_provider)
    assert r.score is None


@pytest.mark.asyncio
async def test_context_sufficiency(mock_provider, sample_question, sample_contexts):
    r = await ContextSufficiencyMetric().compute(
        sample_question, "a", sample_contexts, mock_provider
    )
    assert r.score == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_context_sufficiency_empty(mock_provider):
    r = await ContextSufficiencyMetric().compute("q", "a", [], mock_provider)
    assert r.score == 0.0


@pytest.mark.asyncio
async def test_answer_completeness(mock_provider, sample_question, sample_answer, sample_contexts):
    r = await AnswerCompletenessMetric().compute(
        sample_question, sample_answer, sample_contexts, mock_provider
    )
    assert r.score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_answer_completeness_partial(mock_provider, sample_question, sample_contexts):
    mock_provider.set_response(
        "Break the question into its distinct information needs",
        {"requirements": [
            {"need": "drug name", "addressed": True},
            {"need": "dose", "addressed": False},
        ], "reasoning": "dose missing"},
    )
    r = await AnswerCompletenessMetric().compute(
        sample_question, "Metformin.", sample_contexts, mock_provider
    )
    assert r.score == pytest.approx(0.5)
    assert any("dose" in v for v in r.violations)


@pytest.mark.asyncio
async def test_coherence_and_fluency_normalize_1_to_5(mock_provider, sample_answer):
    for cls, marker in (
        (CoherenceMetric, "Rate the coherence of the answer"),
        (FluencyMetric, "Rate the fluency of the answer"),
    ):
        mock_provider.set_response(marker, {"score": 1, "issues": ["x"], "reasoning": "poor"})
        low = await cls().compute("q", sample_answer, [], mock_provider)
        assert low.score == 0.0
        mock_provider.set_response(marker, {"score": 5, "issues": [], "reasoning": "great"})
        high = await cls().compute("q", sample_answer, [], mock_provider)
        assert high.score == 1.0


@pytest.mark.asyncio
async def test_conciseness_flags_padding(mock_provider, sample_question):
    mock_provider.set_response(
        "Rate how concise the answer is",
        {"conciseness_score": 0.3, "redundant_spans": ["As I mentioned before"],
         "reasoning": "repetitive"},
    )
    r = await ConcisenessMetric().compute(sample_question, "verbose answer", [], mock_provider)
    assert r.score == pytest.approx(0.3)
    assert r.violations


@pytest.mark.asyncio
async def test_refusal_appropriateness_over_refusal(mock_provider, sample_question, sample_contexts):
    mock_provider.set_response(
        "either answered the question or declined",
        {"did_refuse": True, "context_supports_answer": True,
         "appropriate": False, "reasoning": "context had the answer"},
    )
    r = await RefusalAppropriatenessMetric().compute(
        sample_question, "I cannot find this in the context.", sample_contexts, mock_provider
    )
    assert r.score == 0.0
    assert "Over-refusal" in r.violations[0]


@pytest.mark.asyncio
async def test_refusal_appropriateness_correct_answer(mock_provider, sample_question, sample_answer, sample_contexts):
    r = await RefusalAppropriatenessMetric().compute(
        sample_question, sample_answer, sample_contexts, mock_provider
    )
    assert r.score == 1.0


@pytest.mark.asyncio
async def test_citation_support(mock_provider, sample_question, sample_contexts):
    mock_provider.set_response(
        "quoted phrases, figures, or references attributed",
        {"citations": [
            {"span": "500mg twice daily", "supported": True},
            {"span": "per WHO", "supported": False},
        ]},
    )
    r = await CitationSupportMetric().compute(
        sample_question, 'The dose is "500mg twice daily" per WHO.', sample_contexts, mock_provider
    )
    assert r.score == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_citation_support_no_citations(mock_provider, sample_question, sample_contexts):
    r = await CitationSupportMetric().compute(
        sample_question, "A plain answer.", sample_contexts, mock_provider
    )
    assert r.score is None


@pytest.mark.asyncio
async def test_pii_leakage_detected(mock_provider, sample_question):
    mock_provider.set_response(
        "personally identifiable information (PII)",
        {"pii_detected": True,
         "pii_items": [{"type": "phone", "excerpt": "555-0100", "severity": "high"}],
         "leakage_score": 0.9, "reasoning": "phone number exposed"},
    )
    r = await PIILeakageMetric().compute(
        sample_question, "Call the patient at 555-0100.", [], mock_provider
    )
    assert r.score == pytest.approx(0.1)
    assert r.violations


@pytest.mark.asyncio
async def test_pii_leakage_clean(mock_provider, sample_question, sample_answer):
    r = await PIILeakageMetric().compute(sample_question, sample_answer, [], mock_provider)
    assert r.score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_tone_professionalism(mock_provider, sample_question, sample_answer):
    r = await ToneProfessionalismMetric().compute(
        sample_question, sample_answer, [], mock_provider
    )
    assert r.score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_conversation_relevancy(mock_provider, sample_turns):
    r = await ConversationRelevancyMetric().compute(
        "", "", [], mock_provider, turns=sample_turns
    )
    assert r.score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_conversation_relevancy_needs_turns(mock_provider):
    r = await ConversationRelevancyMetric().compute("", "", [], mock_provider)
    assert r.score is None


def test_registry_has_50_metrics_total():
    from ragval.domains import DOMAIN_REGISTRY
    from ragval.metrics.registry import METRIC_REGISTRY

    domain_count = sum(
        len(d.additional_metric_names) for d in DOMAIN_REGISTRY.values()
    )
    assert len(METRIC_REGISTRY) == 40
    assert len(METRIC_REGISTRY) + domain_count == 50
