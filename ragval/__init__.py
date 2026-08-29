"""ragval - The complete RAG evaluation library. Scores, diagnoses, and domain-aware."""

from __future__ import annotations

__version__ = "0.0.1"

from ragval.diagnosis import DiagnosisEngine, DiagnosisResult
from ragval.evaluator import (
    DEFAULT_METRICS,
    FULL_METRICS,
    RAGEvaluator,
    evaluate,
)
from ragval.exceptions import (
    DomainNotFoundError,
    GroundTruthRequiredError,
    InvalidInputError,
    MetricComputationError,
    ProviderError,
    RAGEvalError,
)
from ragval.metrics import get_metric, list_metrics
from ragval.metrics.base import BaseMetric, MetricResult
from ragval.metrics.custom import AspectCritic, GEval, RubricEval
from ragval.providers import LiteLLMProvider, get_provider
from ragval.result import (
    AgentEvaluationResult,
    BatchEvaluationResult,
    ConversationEvaluationResult,
    EvaluationResult,
)

__all__ = [
    "__version__",
    "evaluate",
    "RAGEvaluator",
    "DEFAULT_METRICS",
    "FULL_METRICS",
    "EvaluationResult",
    "AgentEvaluationResult",
    "ConversationEvaluationResult",
    "BatchEvaluationResult",
    "DiagnosisEngine",
    "DiagnosisResult",
    "BaseMetric",
    "MetricResult",
    "get_metric",
    "list_metrics",
    "GEval",
    "RubricEval",
    "AspectCritic",
    "LiteLLMProvider",
    "get_provider",
    "RAGEvalError",
    "ProviderError",
    "MetricComputationError",
    "DomainNotFoundError",
    "GroundTruthRequiredError",
    "InvalidInputError",
]
