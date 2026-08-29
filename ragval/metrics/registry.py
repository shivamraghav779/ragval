"""Central registry mapping metric-name strings to metric classes."""

from __future__ import annotations

from typing import Dict, List, Optional, Type

from ragval.exceptions import DomainNotFoundError
from ragval.metrics.agentic import (
    AgentGoalAccuracyMetric,
    ArgumentCorrectnessMetric,
    PlanAdherenceMetric,
    PlanQualityMetric,
    StepEfficiencyMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
)
from ragval.metrics.base import BaseMetric
from ragval.metrics.conversation import (
    ConversationCompletenessMetric,
    ConversationRelevancyMetric,
    KnowledgeRetentionMetric,
    RoleAdherenceMetric,
)
from ragval.metrics.generation import (
    AnswerCompletenessMetric,
    AnswerCorrectnessMetric,
    AnswerRelevanceMetric,
    AnswerSemanticSimilarityMetric,
    CitationSupportMetric,
    CoherenceMetric,
    ConcisenessMetric,
    FactualCorrectnessMetric,
    FaithfulnessMetric,
    FluencyMetric,
    HallucinationMetric,
    RefusalAppropriatenessMetric,
    SummarizationMetric,
)
from ragval.metrics.retrieval import (
    ContextEntityRecallMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    ContextRelevanceMetric,
    ContextSufficiencyMetric,
    ContextUtilizationMetric,
    HitRateMetric,
    MRRMetric,
    NDCGMetric,
    NoiseSensitivityMetric,
    RetrievalDiversityMetric,
)
from ragval.metrics.safety import (
    BiasMetric,
    PIILeakageMetric,
    ToneProfessionalismMetric,
    TopicAdherenceMetric,
    ToxicityMetric,
)

METRIC_REGISTRY: Dict[str, Type[BaseMetric]] = {
    # retrieval
    "context_precision": ContextPrecisionMetric,
    "context_recall": ContextRecallMetric,
    "context_relevance": ContextRelevanceMetric,
    "context_entity_recall": ContextEntityRecallMetric,
    "noise_sensitivity": NoiseSensitivityMetric,
    "mrr": MRRMetric,
    "ndcg": NDCGMetric,
    "hit_rate": HitRateMetric,
    "context_utilization": ContextUtilizationMetric,
    "retrieval_diversity": RetrievalDiversityMetric,
    "context_sufficiency": ContextSufficiencyMetric,
    # generation
    "faithfulness": FaithfulnessMetric,
    "answer_relevance": AnswerRelevanceMetric,
    "answer_correctness": AnswerCorrectnessMetric,
    "hallucination": HallucinationMetric,
    "factual_correctness": FactualCorrectnessMetric,
    "answer_semantic_similarity": AnswerSemanticSimilarityMetric,
    "summarization": SummarizationMetric,
    "answer_completeness": AnswerCompletenessMetric,
    "coherence": CoherenceMetric,
    "fluency": FluencyMetric,
    "conciseness": ConcisenessMetric,
    "refusal_appropriateness": RefusalAppropriatenessMetric,
    "citation_support": CitationSupportMetric,
    # safety
    "bias": BiasMetric,
    "toxicity": ToxicityMetric,
    "topic_adherence": TopicAdherenceMetric,
    "pii_leakage": PIILeakageMetric,
    "tone_professionalism": ToneProfessionalismMetric,
    # agentic
    "tool_correctness": ToolCorrectnessMetric,
    "argument_correctness": ArgumentCorrectnessMetric,
    "task_completion": TaskCompletionMetric,
    "step_efficiency": StepEfficiencyMetric,
    "plan_adherence": PlanAdherenceMetric,
    "plan_quality": PlanQualityMetric,
    "agent_goal_accuracy": AgentGoalAccuracyMetric,
    # conversation
    "conversation_completeness": ConversationCompletenessMetric,
    "knowledge_retention": KnowledgeRetentionMetric,
    "role_adherence": RoleAdherenceMetric,
    "conversation_relevancy": ConversationRelevancyMetric,
}


def get_metric(name: str) -> BaseMetric:
    """Return a fresh instance of the metric registered under ``name``."""
    try:
        return METRIC_REGISTRY[name]()
    except KeyError:
        raise DomainNotFoundError(name, list(METRIC_REGISTRY)) from None


def list_metrics(category: Optional[str] = None) -> List[str]:
    """All registered metric names, optionally filtered by category."""
    if category is None:
        return list(METRIC_REGISTRY)
    return [
        name
        for name, cls in METRIC_REGISTRY.items()
        if getattr(cls, "category", None) == category
    ]
