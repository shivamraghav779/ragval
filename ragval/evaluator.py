"""RAGEvaluator: the reusable entry point for every evaluation mode."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from ragval.batch import run_batch
from ragval.diagnosis import DiagnosisEngine
from ragval.domains import get_domain
from ragval.exceptions import InvalidInputError
from ragval.metrics.base import MetricResult
from ragval.metrics.generation.faithfulness import FaithfulnessMetric
from ragval.metrics.registry import METRIC_REGISTRY, get_metric
from ragval.providers import get_provider
from ragval.result import (
    AgentEvaluationResult,
    BatchEvaluationResult,
    ConversationEvaluationResult,
    EvaluationResult,
)
from ragval.utils.async_utils import run_sync

DEFAULT_METRICS = [
    "faithfulness",
    "context_precision",
    "answer_relevance",
    "hallucination",
]
FULL_METRICS = list(METRIC_REGISTRY)

_STANDARD_CATEGORIES = {"retrieval", "generation", "safety"}
_DEPENDENT = {"noise_sensitivity"}


def _category(name: str) -> str:
    cls = METRIC_REGISTRY.get(name)
    return getattr(cls, "category", "generation") if cls else "generation"


class RAGEvaluator:
    def __init__(
        self,
        model: str = "groq/llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        domain: str = "general",
        faithfulness_threshold: float = 0.8,
        warn_threshold: float = 0.5,
        metrics: Optional[List[str]] = None,
        run_domain_metrics: bool = True,
        run_diagnosis: bool = True,
        timeout: int = 30,
        max_tokens: int = 1000,
        max_concurrency: int = 8,
    ) -> None:
        self.model = model
        self.provider = get_provider(
            model,
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_tokens=max_tokens,
            max_concurrency=max_concurrency,
        )
        self.domain_name = domain
        self.domain = get_domain(domain)
        self.faithfulness_threshold = faithfulness_threshold
        self.warn_threshold = warn_threshold
        self.run_domain_metrics = run_domain_metrics
        self.run_diagnosis = run_diagnosis
        self._diagnosis_engine = DiagnosisEngine()

        if metrics is None:
            selected = list(DEFAULT_METRICS)
        elif isinstance(metrics, str):
            selected = FULL_METRICS if metrics == "all" else [metrics]
        elif list(metrics) == ["all"]:
            selected = list(FULL_METRICS)
        else:
            selected = list(metrics)

        # Validate names now so typos fail fast.
        for name in selected:
            if name not in METRIC_REGISTRY:
                raise InvalidInputError(
                    "metrics", f"unknown metric {name!r}. See ragval list-metrics."
                )
        self.metric_names = selected

    # ------------------------------------------------------------------
    # single evaluation
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        start = time.perf_counter()
        contexts = list(contexts or [])
        kwargs.setdefault("domain", self.domain_name)

        standard = [
            n
            for n in self.metric_names
            if _category(n) in _STANDARD_CATEGORIES and n not in _DEPENDENT
        ]
        dependent = [n for n in self.metric_names if n in _DEPENDENT]

        # Independent metrics + domain metrics, all in parallel.
        metric_objs = [get_metric(n) for n in standard]
        tasks = [
            m.compute(question, answer, contexts, self.provider, ground_truth, **kwargs)
            for m in metric_objs
        ]

        domain_task = None
        if self.run_domain_metrics and self.domain_name != "general":
            domain_task = asyncio.ensure_future(
                self.domain.get_domain_metrics(
                    question, answer, contexts, self.provider
                )
            )

        results = await asyncio.gather(*tasks) if tasks else []
        metric_results: Dict[str, MetricResult] = {
            name: res for name, res in zip(standard, results)
        }

        # Dependent metrics (noise_sensitivity needs faithfulness).
        for name in dependent:
            extra = dict(kwargs)
            if "faithfulness" in metric_results and metric_results["faithfulness"].score is not None:
                extra["clean_faithfulness_score"] = metric_results["faithfulness"].score
            metric_results[name] = await get_metric(name).compute(
                question, answer, contexts, self.provider, ground_truth, **extra
            )

        domain_metrics: Dict[str, Optional[MetricResult]] = {}
        if domain_task is not None:
            domain_metrics = await domain_task

        result = self._assemble(
            EvaluationResult,
            question,
            answer,
            contexts,
            ground_truth,
            metric_results,
            domain_metrics,
        )
        result.total_duration_ms = (time.perf_counter() - start) * 1000.0
        return result

    def evaluate_sync(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        return run_sync(
            self.evaluate(question, answer, contexts, ground_truth, **kwargs)
        )

    # ------------------------------------------------------------------
    # pipeline evaluation
    # ------------------------------------------------------------------

    async def evaluate_pipeline(
        self,
        question: str,
        answer: str,
        retrieved_chunks: List[Any],
        ground_truth: Optional[str] = None,
        expected_chunks: Optional[List[Any]] = None,
        k: int = 5,
        **kwargs: Any,
    ) -> EvaluationResult:
        contexts = [_chunk_text(c) for c in retrieved_chunks or []]
        kwargs.setdefault("k", k)

        # Ensure the ranking metrics run when we have expectations to compare to.
        extra_metrics = []
        if expected_chunks is not None:
            kwargs.setdefault("expected_chunks", [_chunk_text(c) for c in expected_chunks])
            for m in ("mrr", "ndcg", "hit_rate", "context_recall"):
                if m not in self.metric_names:
                    extra_metrics.append(m)

        original = self.metric_names
        try:
            self.metric_names = list(dict.fromkeys(original + extra_metrics))
            return await self.evaluate(
                question, answer, contexts, ground_truth, **kwargs
            )
        finally:
            self.metric_names = original

    # ------------------------------------------------------------------
    # agent evaluation
    # ------------------------------------------------------------------

    async def evaluate_agent(
        self,
        question: str,
        answer: str,
        tool_calls: Optional[List[Any]] = None,
        expected_tools: Optional[List[Any]] = None,
        action_trace: Optional[List[Any]] = None,
        declared_plan: Optional[str] = None,
        ground_truth: Optional[str] = None,
        contexts: Optional[List[str]] = None,
        expected_tool_calls: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> AgentEvaluationResult:
        start = time.perf_counter()
        contexts = list(contexts or [])
        agent_kwargs = {
            "tool_calls": tool_calls,
            "expected_tools": expected_tools,
            "expected_tool_calls": expected_tool_calls or expected_tools,
            "action_trace": action_trace,
            "declared_plan": declared_plan,
            **kwargs,
        }

        # Standard metrics.
        base_result = await self.evaluate(
            question, answer, contexts, ground_truth, **kwargs
        )

        agentic_names = [
            n for n in METRIC_REGISTRY if _category(n) == "agentic"
        ]
        agentic_results = await asyncio.gather(
            *(
                get_metric(n).compute(
                    question, answer, contexts, self.provider, ground_truth,
                    **agent_kwargs,
                )
                for n in agentic_names
            )
        )

        agent_result = AgentEvaluationResult(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
            domain=self.domain_name,
            model=self.model,
            domain_metrics=base_result.domain_metrics,
            pass_threshold=self.faithfulness_threshold,
            warn_threshold=self.warn_threshold,
        )
        from ragval.result import (
            _GENERATION_FIELDS,
            _RETRIEVAL_FIELDS,
            _SAFETY_FIELDS,
        )

        for name in _RETRIEVAL_FIELDS + _GENERATION_FIELDS + _SAFETY_FIELDS:
            setattr(agent_result, name, getattr(base_result, name, None))
        for name, res in zip(agentic_names, agentic_results):
            setattr(agent_result, name, res)

        if self.run_diagnosis:
            agent_result.diagnosis = self._diagnosis_engine.analyze(
                agent_result.all_metrics()
            )
        agent_result.total_duration_ms = (time.perf_counter() - start) * 1000.0
        return agent_result

    # ------------------------------------------------------------------
    # conversation evaluation
    # ------------------------------------------------------------------

    async def evaluate_conversation(
        self,
        turns: List[Dict[str, Any]],
        system_role: Optional[str] = None,
        **kwargs: Any,
    ) -> ConversationEvaluationResult:
        start = time.perf_counter()
        turns = list(turns or [])
        conv_kwargs = {"turns": turns, "system_role": system_role, **kwargs}

        conv_names = [
            "conversation_completeness",
            "knowledge_retention",
            "role_adherence",
            "conversation_relevancy",
        ]
        conv_results = await asyncio.gather(
            *(
                get_metric(n).compute("", "", [], self.provider, None, **conv_kwargs)
                for n in conv_names
            )
        )

        # Per-assistant-turn faithfulness where that turn carries contexts.
        faith = FaithfulnessMetric()
        per_turn: List[Optional[MetricResult]] = []
        for i, turn in enumerate(turns):
            if turn.get("role") != "assistant":
                continue
            ctxs = turn.get("retrieved_contexts") or turn.get("contexts")
            if not ctxs:
                per_turn.append(None)
                continue
            prev_user = ""
            for j in range(i - 1, -1, -1):
                if turns[j].get("role") == "user":
                    prev_user = turns[j].get("content", "")
                    break
            per_turn.append(
                await faith.compute(
                    prev_user, turn.get("content", ""), list(ctxs), self.provider
                )
            )

        result = ConversationEvaluationResult(
            turn_count=len(turns),
            system_role=system_role,
            conversation_completeness=conv_results[0],
            knowledge_retention=conv_results[1],
            role_adherence=conv_results[2],
            conversation_relevancy=conv_results[3],
            per_turn_faithfulness=per_turn,
            model=self.model,
            domain=self.domain_name,
            warn_threshold=self.warn_threshold,
            pass_threshold=self.faithfulness_threshold,
        )
        result.total_duration_ms = (time.perf_counter() - start) * 1000.0
        return result

    # ------------------------------------------------------------------
    # batch
    # ------------------------------------------------------------------

    async def batch_evaluate(
        self,
        qa_pairs: List[Dict[str, Any]],
        concurrency: int = 3,
        **kwargs: Any,
    ) -> BatchEvaluationResult:
        return await run_batch(self, qa_pairs, concurrency, **kwargs)

    def batch_evaluate_sync(
        self,
        qa_pairs: List[Dict[str, Any]],
        concurrency: int = 3,
        **kwargs: Any,
    ) -> BatchEvaluationResult:
        return run_sync(self.batch_evaluate(qa_pairs, concurrency, **kwargs))

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _assemble(
        self,
        cls,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str],
        metric_results: Dict[str, MetricResult],
        domain_metrics: Dict[str, Optional[MetricResult]],
    ) -> EvaluationResult:
        result = cls(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
            domain=self.domain_name,
            model=self.model,
            domain_metrics=domain_metrics or {},
            pass_threshold=self.faithfulness_threshold,
            warn_threshold=self.warn_threshold,
        )
        for name, mr in metric_results.items():
            if hasattr(result, name):
                setattr(result, name, mr)

        if self.run_diagnosis:
            result.diagnosis = self._diagnosis_engine.analyze(result.all_metrics())
        return result

    def _build_error_result(
        self, pair: Dict[str, Any], message: str
    ) -> EvaluationResult:
        result = EvaluationResult(
            question=pair.get("question", ""),
            answer=pair.get("answer", ""),
            contexts=list(pair.get("contexts", [])),
            ground_truth=pair.get("ground_truth"),
            domain=self.domain_name,
            model=self.model,
            pass_threshold=self.faithfulness_threshold,
            warn_threshold=self.warn_threshold,
        )
        result.faithfulness = MetricResult.error("faithfulness", message)
        if self.run_diagnosis:
            result.diagnosis = self._diagnosis_engine.analyze(result.all_metrics())
        return result


def _chunk_text(chunk: Any) -> str:
    if isinstance(chunk, str):
        return chunk
    for attr in ("text", "content", "page_content"):
        if hasattr(chunk, attr):
            return getattr(chunk, attr)
    if isinstance(chunk, dict):
        return chunk.get("text") or chunk.get("content") or str(chunk)
    return str(chunk)


# ----------------------------------------------------------------------
# module-level convenience
# ----------------------------------------------------------------------

async def evaluate(
    question: str,
    answer: str,
    contexts: List[str],
    model: str = "groq/llama-3.3-70b-versatile",
    domain: str = "general",
    ground_truth: Optional[str] = None,
    **kwargs: Any,
) -> EvaluationResult:
    """One-shot evaluation with a fresh evaluator."""
    evaluator = RAGEvaluator(model=model, domain=domain)
    return await evaluator.evaluate(
        question, answer, contexts, ground_truth, **kwargs
    )
