"""Diagnosis decision-tree behaviour."""

from __future__ import annotations

from ragval.diagnosis import DiagnosisEngine
from ragval.metrics.base import MetricResult


def _mr(name: str, score, **metadata) -> MetricResult:
    return MetricResult(
        score=score, reasoning="", violations=[], metadata=metadata, metric_name=name
    )


ENGINE = DiagnosisEngine()


def test_generation_hallucination_despite_good_retrieval():
    scores = {
        "hallucination": _mr("hallucination", 0.4, hallucination_detected=True),
        "context_precision": _mr("context_precision", 0.8),
    }
    d = ENGINE.analyze(scores)
    assert d.failed_layer == "generation"
    assert d.confidence == 0.90


def test_low_faithfulness_good_retrieval():
    scores = {
        "faithfulness": _mr("faithfulness", 0.4),
        "context_precision": _mr("context_precision", 0.8),
        "hallucination": _mr("hallucination", 1.0, hallucination_detected=False),
    }
    d = ENGINE.analyze(scores)
    assert d.failed_layer == "generation"
    assert d.confidence == 0.85


def test_low_context_precision_is_retrieval():
    scores = {"context_precision": _mr("context_precision", 0.3)}
    d = ENGINE.analyze(scores)
    assert d.failed_layer == "retrieval"
    assert d.confidence == 0.82


def test_low_context_recall_is_retrieval():
    scores = {
        "context_precision": _mr("context_precision", 0.9),
        "context_recall": _mr("context_recall", 0.3),
    }
    d = ENGINE.analyze(scores)
    assert d.failed_layer == "retrieval"
    assert d.confidence == 0.80


def test_grounded_but_off_question_is_prompt():
    scores = {
        "faithfulness": _mr("faithfulness", 0.9),
        "answer_relevance": _mr("answer_relevance", 0.3),
        "context_precision": _mr("context_precision", 0.8),
    }
    d = ENGINE.analyze(scores)
    assert d.failed_layer == "prompt"
    assert d.confidence == 0.83


def test_noise_sensitive_is_fusion_ranking():
    scores = {
        "noise_sensitivity": _mr("noise_sensitivity", 0.3),
        "faithfulness": _mr("faithfulness", 0.8),
        "context_precision": _mr("context_precision", 0.8),
        "answer_relevance": _mr("answer_relevance", 0.8),
    }
    d = ENGINE.analyze(scores)
    assert d.failed_layer == "fusion_ranking"
    assert d.confidence == 0.74


def test_knowledge_base_wrong():
    scores = {
        "faithfulness": _mr("faithfulness", 0.9),
        "context_precision": _mr("context_precision", 0.8),
        "answer_correctness": _mr("answer_correctness", 0.3),
    }
    d = ENGINE.analyze(scores)
    assert d.failed_layer == "knowledge_base"
    assert d.confidence == 0.77


def test_all_good_no_failed_layer():
    scores = {
        "faithfulness": _mr("faithfulness", 0.9),
        "context_precision": _mr("context_precision", 0.85),
        "answer_relevance": _mr("answer_relevance", 0.9),
        "hallucination": _mr("hallucination", 1.0, hallucination_detected=False),
    }
    d = ENGINE.analyze(scores)
    assert d.failed_layer is None
    assert d.confidence == 0.65


def test_secondary_issues_detected_alongside_primary():
    scores = {
        "hallucination": _mr("hallucination", 0.4, hallucination_detected=True),
        "context_precision": _mr("context_precision", 0.8),
        "faithfulness": _mr("faithfulness", 0.9),
        "answer_relevance": _mr("answer_relevance", 0.3),
    }
    d = ENGINE.analyze(scores)
    assert d.failed_layer == "generation"  # condition 1 wins
    assert any("prompt" in s for s in d.secondary_issues)
