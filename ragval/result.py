"""Result containers returned by the evaluator.

``MetricResult`` lives in :mod:`ragval.metrics.base` and is re-exported here.
``DiagnosisResult`` is defined here and re-exported from :mod:`ragval.diagnosis`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ragval.metrics.base import MetricResult

__all__ = [
    "MetricResult",
    "DiagnosisResult",
    "EvaluationResult",
    "AgentEvaluationResult",
    "ConversationEvaluationResult",
    "BatchEvaluationResult",
]

# Core metrics and their weights for the blended overall score.
_CORE_WEIGHTS: Dict[str, float] = {
    "faithfulness": 0.30,
    "context_precision": 0.20,
    "answer_relevance": 0.20,
    "hallucination": 0.15,
    "context_recall": 0.10,
    "answer_correctness": 0.05,
}

_RETRIEVAL_FIELDS = [
    "context_precision", "context_recall", "context_relevance",
    "context_entity_recall", "noise_sensitivity", "mrr", "ndcg", "hit_rate",
    "context_utilization", "retrieval_diversity", "context_sufficiency",
]
_GENERATION_FIELDS = [
    "faithfulness", "answer_relevance", "answer_correctness", "hallucination",
    "factual_correctness", "answer_semantic_similarity", "summarization",
    "answer_completeness", "coherence", "fluency", "conciseness",
    "refusal_appropriateness", "citation_support",
]
_SAFETY_FIELDS = [
    "bias", "toxicity", "topic_adherence", "pii_leakage", "tone_professionalism",
]
_AGENTIC_FIELDS = [
    "tool_correctness", "argument_correctness", "task_completion",
    "step_efficiency", "plan_adherence", "plan_quality", "agent_goal_accuracy",
]


@dataclass
class DiagnosisResult:
    failed_layer: Optional[str]
    root_cause: str
    suggested_fix: str
    confidence: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    secondary_issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failed_layer": self.failed_layer,
            "root_cause": self.root_cause,
            "suggested_fix": self.suggested_fix,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "secondary_issues": list(self.secondary_issues),
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _score_of(mr: Optional[MetricResult]) -> Optional[float]:
    return mr.score if mr is not None else None


@dataclass
class EvaluationResult:
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None
    domain: str = "general"
    model: str = ""

    # retrieval
    context_precision: Optional[MetricResult] = None
    context_recall: Optional[MetricResult] = None
    context_relevance: Optional[MetricResult] = None
    context_entity_recall: Optional[MetricResult] = None
    noise_sensitivity: Optional[MetricResult] = None
    mrr: Optional[MetricResult] = None
    ndcg: Optional[MetricResult] = None
    hit_rate: Optional[MetricResult] = None
    context_utilization: Optional[MetricResult] = None
    retrieval_diversity: Optional[MetricResult] = None
    context_sufficiency: Optional[MetricResult] = None

    # generation
    faithfulness: Optional[MetricResult] = None
    answer_relevance: Optional[MetricResult] = None
    answer_correctness: Optional[MetricResult] = None
    hallucination: Optional[MetricResult] = None
    factual_correctness: Optional[MetricResult] = None
    answer_semantic_similarity: Optional[MetricResult] = None
    summarization: Optional[MetricResult] = None
    answer_completeness: Optional[MetricResult] = None
    coherence: Optional[MetricResult] = None
    fluency: Optional[MetricResult] = None
    conciseness: Optional[MetricResult] = None
    refusal_appropriateness: Optional[MetricResult] = None
    citation_support: Optional[MetricResult] = None

    # safety
    bias: Optional[MetricResult] = None
    toxicity: Optional[MetricResult] = None
    topic_adherence: Optional[MetricResult] = None
    pii_leakage: Optional[MetricResult] = None
    tone_professionalism: Optional[MetricResult] = None

    domain_metrics: Dict[str, Optional[MetricResult]] = field(default_factory=dict)
    diagnosis: Optional[DiagnosisResult] = None

    pass_threshold: float = 0.8
    warn_threshold: float = 0.5
    _evaluated_at: str = field(default_factory=_now_iso)
    total_duration_ms: float = 0.0

    # -- metric access helpers ------------------------------------------

    _METRIC_FIELDS = _RETRIEVAL_FIELDS + _GENERATION_FIELDS + _SAFETY_FIELDS

    def all_metrics(self) -> Dict[str, MetricResult]:
        out: Dict[str, MetricResult] = {}
        for name in self._METRIC_FIELDS:
            mr = getattr(self, name, None)
            if mr is not None:
                out[name] = mr
        for name, mr in (self.domain_metrics or {}).items():
            if mr is not None:
                out[name] = mr
        return out

    # -- computed properties ------------------------------------------

    @property
    def hallucination_detected(self) -> bool:
        if self.hallucination is None:
            return False
        return bool(self.hallucination.metadata.get("hallucination_detected"))

    @property
    def overall_score(self) -> float:
        scores: List[Optional[float]] = []
        weights: List[float] = []
        for name, weight in _CORE_WEIGHTS.items():
            mr = getattr(self, name, None)
            s = _score_of(mr)
            if s is None:
                continue
            scores.append(s)
            weights.append(weight)
        if not scores:
            return 0.0
        total_w = sum(weights)
        return sum(s * w for s, w in zip(scores, weights)) / total_w

    @property
    def verdict(self) -> str:
        overall = self.overall_score
        faith = _score_of(self.faithfulness)

        if self.hallucination_detected:
            return "FAIL"
        if faith is not None and faith < self.warn_threshold:
            return "FAIL"
        if overall < self.warn_threshold:
            return "FAIL"

        if (
            overall >= self.pass_threshold
            and not self.hallucination_detected
            and (faith is None or faith >= self.pass_threshold)
        ):
            return "PASS"
        return "WARN"

    @property
    def evaluated_at(self) -> str:
        return self._evaluated_at

    # -- serialization ------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "contexts": list(self.contexts),
            "ground_truth": self.ground_truth,
            "domain": self.domain,
            "model": self.model,
            "verdict": self.verdict,
            "overall_score": round(self.overall_score, 4),
            "hallucination_detected": self.hallucination_detected,
            "evaluated_at": self.evaluated_at,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "metrics": {
                name: mr.to_dict() for name, mr in self.all_metrics().items()
            },
            "domain_metrics": {
                name: (mr.to_dict() if mr else None)
                for name, mr in (self.domain_metrics or {}).items()
            },
            "diagnosis": self.diagnosis.to_dict() if self.diagnosis else None,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> str:
        return (
            f"[{self.verdict}] overall={self.overall_score:.2f} "
            f"domain={self.domain} model={self.model} "
            f"({len(self.all_metrics())} metrics)"
        )

    def __str__(self) -> str:
        lines = []
        width = 68
        lines.append("+" + "-" * width + "+")
        lines.append(f"| ragval  ::  verdict = {self.verdict:<45}|")
        lines.append(f"| overall score: {self.overall_score:.3f}"
                     f"{' ':>{width - 21}}|")
        lines.append("+" + "-" * width + "+")
        for name, mr in self.all_metrics().items():
            s = "n/a" if mr.score is None else f"{mr.score:.3f}"
            lines.append(f"| {name:<34} {s:>8}"
                         f"{' ':>{width - 45}}|")
        if self.diagnosis:
            lines.append("+" + "-" * width + "+")
            lines.append(f"| diagnosis: failed_layer = "
                         f"{str(self.diagnosis.failed_layer):<38}|")
            for chunk in _wrap(self.diagnosis.suggested_fix, width - 4):
                lines.append(f"|   {chunk:<{width - 3}}|")
        lines.append("+" + "-" * width + "+")
        return "\n".join(lines)


def _wrap(text: str, width: int) -> List[str]:
    words = text.split()
    out: List[str] = []
    current = ""
    for w in words:
        if len(current) + len(w) + 1 > width:
            out.append(current)
            current = w
        else:
            current = f"{current} {w}".strip()
    if current:
        out.append(current)
    return out or [""]


@dataclass
class AgentEvaluationResult(EvaluationResult):
    tool_correctness: Optional[MetricResult] = None
    argument_correctness: Optional[MetricResult] = None
    task_completion: Optional[MetricResult] = None
    step_efficiency: Optional[MetricResult] = None
    plan_adherence: Optional[MetricResult] = None
    plan_quality: Optional[MetricResult] = None
    agent_goal_accuracy: Optional[MetricResult] = None

    _METRIC_FIELDS = (
        _RETRIEVAL_FIELDS + _GENERATION_FIELDS + _SAFETY_FIELDS + _AGENTIC_FIELDS
    )


@dataclass
class ConversationEvaluationResult:
    turn_count: int
    system_role: Optional[str] = None
    conversation_completeness: Optional[MetricResult] = None
    knowledge_retention: Optional[MetricResult] = None
    role_adherence: Optional[MetricResult] = None
    conversation_relevancy: Optional[MetricResult] = None
    per_turn_faithfulness: List[Optional[MetricResult]] = field(default_factory=list)
    model: str = ""
    domain: str = "general"
    _evaluated_at: str = field(default_factory=_now_iso)
    total_duration_ms: float = 0.0
    warn_threshold: float = 0.5
    pass_threshold: float = 0.8

    def _metrics(self) -> Dict[str, MetricResult]:
        out = {}
        for name in (
            "conversation_completeness",
            "knowledge_retention",
            "role_adherence",
            "conversation_relevancy",
        ):
            mr = getattr(self, name)
            if mr is not None:
                out[name] = mr
        return out

    @property
    def overall_conversation_score(self) -> float:
        scores = [mr.score for mr in self._metrics().values() if mr.score is not None]
        turn_scores = [
            mr.score for mr in self.per_turn_faithfulness if mr and mr.score is not None
        ]
        all_scores = scores + turn_scores
        return sum(all_scores) / len(all_scores) if all_scores else 0.0

    @property
    def verdict(self) -> str:
        s = self.overall_conversation_score
        if s < self.warn_threshold:
            return "FAIL"
        if s >= self.pass_threshold:
            return "PASS"
        return "WARN"

    @property
    def evaluated_at(self) -> str:
        return self._evaluated_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_count": self.turn_count,
            "system_role": self.system_role,
            "model": self.model,
            "domain": self.domain,
            "verdict": self.verdict,
            "overall_conversation_score": round(self.overall_conversation_score, 4),
            "evaluated_at": self.evaluated_at,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "metrics": {n: mr.to_dict() for n, mr in self._metrics().items()},
            "per_turn_faithfulness": [
                (mr.to_dict() if mr else None) for mr in self.per_turn_faithfulness
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> str:
        return (
            f"[{self.verdict}] conversation score="
            f"{self.overall_conversation_score:.2f} turns={self.turn_count}"
        )


@dataclass
class BatchEvaluationResult:
    results: List[EvaluationResult]
    domain: str = "general"
    model: str = ""
    _evaluated_at: str = field(default_factory=_now_iso)
    total_duration_ms: float = 0.0

    # populated in __post_init__
    total: int = 0
    pass_count: int = 0
    warn_count: int = 0
    fail_count: int = 0
    avg_faithfulness: float = 0.0
    avg_context_precision: float = 0.0
    avg_answer_relevance: float = 0.0
    hallucination_rate: float = 0.0
    avg_overall_score: float = 0.0
    diagnosis_summary: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.total = len(self.results)
        self.pass_count = sum(1 for r in self.results if r.verdict == "PASS")
        self.warn_count = sum(1 for r in self.results if r.verdict == "WARN")
        self.fail_count = sum(1 for r in self.results if r.verdict == "FAIL")

        self.avg_faithfulness = _avg(self.results, "faithfulness")
        self.avg_context_precision = _avg(self.results, "context_precision")
        self.avg_answer_relevance = _avg(self.results, "answer_relevance")
        self.avg_overall_score = (
            sum(r.overall_score for r in self.results) / self.total
            if self.total
            else 0.0
        )
        self.hallucination_rate = (
            sum(1 for r in self.results if r.hallucination_detected) / self.total
            if self.total
            else 0.0
        )
        self.diagnosis_summary = self._build_diagnosis_summary()

    def _build_diagnosis_summary(self) -> Dict[str, Any]:
        layer_dist: Dict[str, int] = {}
        fix_counts: Dict[str, int] = {}
        secondary: Dict[str, int] = {}
        for r in self.results:
            if not r.diagnosis:
                continue
            layer = r.diagnosis.failed_layer or "none"
            layer_dist[layer] = layer_dist.get(layer, 0) + 1
            if r.diagnosis.failed_layer:
                fix_counts[r.diagnosis.suggested_fix] = (
                    fix_counts.get(r.diagnosis.suggested_fix, 0) + 1
                )
            for issue in r.diagnosis.secondary_issues:
                secondary[issue] = secondary.get(issue, 0) + 1

        most_common_failure = None
        non_none = {k: v for k, v in layer_dist.items() if k != "none"}
        if non_none:
            most_common_failure = max(non_none, key=non_none.get)
        most_common_fix = (
            max(fix_counts, key=fix_counts.get) if fix_counts else None
        )
        return {
            "failed_layer_distribution": layer_dist,
            "most_common_failure": most_common_failure,
            "most_common_suggested_fix": most_common_fix,
            "secondary_issues_frequency": secondary,
        }

    # -- reporting ---------------------------------------------------

    def report(self) -> str:
        lines = [
            "# ragval batch report",
            "",
            f"- model: `{self.model}`",
            f"- domain: `{self.domain}`",
            f"- total: {self.total}",
            f"- PASS / WARN / FAIL: {self.pass_count} / {self.warn_count} / {self.fail_count}",
            f"- avg overall score: {self.avg_overall_score:.3f}",
            f"- avg faithfulness: {self.avg_faithfulness:.3f}",
            f"- avg context precision: {self.avg_context_precision:.3f}",
            f"- avg answer relevance: {self.avg_answer_relevance:.3f}",
            f"- hallucination rate: {self.hallucination_rate:.1%}",
            "",
            "| # | verdict | overall | faithfulness | ctx precision | failed layer |",
            "|---|---------|---------|--------------|---------------|--------------|",
        ]
        for i, r in enumerate(self.results):
            faith = _fmt(_score_of(r.faithfulness))
            cp = _fmt(_score_of(r.context_precision))
            layer = r.diagnosis.failed_layer if r.diagnosis else "-"
            lines.append(
                f"| {i} | {r.verdict} | {r.overall_score:.2f} | {faith} | {cp} "
                f"| {layer or '-'} |"
            )
        ds = self.diagnosis_summary
        lines += [
            "",
            "## diagnosis summary",
            "",
            f"- failed-layer distribution: `{ds['failed_layer_distribution']}`",
            f"- most common failure: `{ds['most_common_failure']}`",
        ]
        if ds["most_common_suggested_fix"]:
            lines.append(f"- most common fix: {ds['most_common_suggested_fix']}")
        return "\n".join(lines)

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "to_dataframe() needs pandas. Install it with: "
                'pip install "ragval[pandas]"'
            ) from exc
        rows = []
        for i, r in enumerate(self.results):
            row = {
                "index": i,
                "verdict": r.verdict,
                "overall_score": r.overall_score,
                "hallucination_detected": r.hallucination_detected,
                "question": r.question,
                "failed_layer": r.diagnosis.failed_layer if r.diagnosis else None,
            }
            for name, mr in r.all_metrics().items():
                row[name] = mr.score
            rows.append(row)
        return pd.DataFrame(rows)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "pass_count": self.pass_count,
            "warn_count": self.warn_count,
            "fail_count": self.fail_count,
            "avg_faithfulness": self.avg_faithfulness,
            "avg_context_precision": self.avg_context_precision,
            "avg_answer_relevance": self.avg_answer_relevance,
            "hallucination_rate": self.hallucination_rate,
            "avg_overall_score": self.avg_overall_score,
            "domain": self.domain,
            "model": self.model,
            "evaluated_at": self._evaluated_at,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "diagnosis_summary": self.diagnosis_summary,
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def worst_cases(self, n: int = 5) -> List[EvaluationResult]:
        return sorted(self.results, key=lambda r: r.overall_score)[:n]

    def best_cases(self, n: int = 5) -> List[EvaluationResult]:
        return sorted(self.results, key=lambda r: r.overall_score, reverse=True)[:n]

    def filter_by_verdict(self, verdict: str) -> List[EvaluationResult]:
        return [r for r in self.results if r.verdict == verdict.upper()]


def _avg(results: List[EvaluationResult], field_name: str) -> float:
    vals = [
        getattr(r, field_name).score
        for r in results
        if getattr(r, field_name, None) is not None
        and getattr(r, field_name).score is not None
    ]
    return sum(vals) / len(vals) if vals else 0.0


def _fmt(score: Optional[float]) -> str:
    return "n/a" if score is None else f"{score:.2f}"
