"""The pipeline-layer diagnosis engine.

Given the metric results for one evaluation, walk an ordered decision tree and
return the single most likely failing layer, its root cause, and a concrete
fix. Remaining matched conditions become ``secondary_issues``.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from ragval.metrics.base import MetricResult
from ragval.result import DiagnosisResult

__all__ = ["DiagnosisEngine", "DiagnosisResult"]


def _score(scores: Dict[str, MetricResult], name: str) -> Optional[float]:
    mr = scores.get(name)
    if mr is None:
        return None
    return mr.score


def _hallucination_detected(scores: Dict[str, MetricResult]) -> bool:
    mr = scores.get("hallucination")
    if mr is None:
        return False
    return bool(mr.metadata.get("hallucination_detected"))


# Each condition: (key, predicate, builder) where builder returns the
# (failed_layer, root_cause, suggested_fix, confidence) tuple.
Condition = Tuple[str, Callable[[Dict[str, MetricResult]], bool], dict]


class DiagnosisEngine:
    """Ordered decision tree. First matching condition is the primary diagnosis."""

    def analyze(self, metric_scores: Dict[str, MetricResult]) -> DiagnosisResult:
        s = metric_scores

        cp = _score(s, "context_precision")
        cr = _score(s, "context_recall")
        faith = _score(s, "faithfulness")
        ar = _score(s, "answer_relevance")
        ns = _score(s, "noise_sensitivity")
        ac = _score(s, "answer_correctness")
        halluc = _hallucination_detected(s)

        conditions: List[Tuple[str, bool, Dict]] = []

        # 1 - Generation hallucination despite good retrieval.
        conditions.append((
            "generation_hallucination",
            halluc and cp is not None and cp > 0.6,
            {
                "failed_layer": "generation",
                "root_cause": (
                    "Relevant context was retrieved but the model fabricated "
                    "specific facts not present in any retrieved chunk. This is a "
                    "hallucination pattern occurring at the generation layer "
                    "despite adequate retrieval."
                ),
                "suggested_fix": (
                    "Use a more instruction-following model. Set temperature to "
                    "0.0 for factual domains. Add explicit grounding instruction: "
                    "'Only state facts that appear verbatim or by clear "
                    "implication in the context. If a specific fact is not in the "
                    "context, do not state it.' Consider adding a post-generation "
                    "entity verification step."
                ),
                "confidence": 0.90,
            },
        ))

        # 2 - Low faithfulness with good retrieval.
        conditions.append((
            "low_faithfulness_good_retrieval",
            faith is not None and faith < 0.5 and cp is not None and cp > 0.6,
            {
                "failed_layer": "generation",
                "root_cause": (
                    "Relevant context was retrieved but the LLM generated claims "
                    "beyond what the context supports. The retrieval layer is "
                    "working but generation is hallucinating."
                ),
                "suggested_fix": (
                    "Strengthen grounding instruction in system prompt. Add: "
                    "'Answer ONLY from the provided context. If the answer is not "
                    "in the context, say so explicitly. Do not infer or "
                    "extrapolate.' Consider a stricter model or lower temperature."
                ),
                "confidence": 0.85,
            },
        ))

        # 3 - Low context precision.
        conditions.append((
            "low_context_precision",
            cp is not None and cp < 0.5,
            {
                "failed_layer": "retrieval",
                "root_cause": (
                    "More than half of retrieved chunks are irrelevant to the "
                    "query. The retriever is injecting noise into the context "
                    "window which confuses the generator and increases "
                    "hallucination risk."
                ),
                "suggested_fix": (
                    "Improve chunking granularity - smaller, more focused chunks "
                    "retrieve more precisely. Add a reranking step (e.g. Cohere "
                    "Rerank) after initial retrieval. Review your embedding model "
                    "- domain-specific models outperform general ones for "
                    "specialized content. Consider hybrid retrieval if using only "
                    "dense search."
                ),
                "confidence": 0.82,
            },
        ))

        # 4 - Low context recall.
        conditions.append((
            "low_context_recall",
            cr is not None and cr < 0.5,
            {
                "failed_layer": "retrieval",
                "root_cause": (
                    "The retriever is missing relevant information that exists in "
                    "the knowledge base. The answer cannot be complete because "
                    "the right chunks were not fetched."
                ),
                "suggested_fix": (
                    "Increase top_k in your retrieval step. Use hybrid retrieval "
                    "(dense + sparse/BM25) if currently using only one method. "
                    "Review chunking boundaries - key information may be split "
                    "across chunks. Consider query expansion or HyDE (hypothetical "
                    "document embeddings) for better query-chunk matching."
                ),
                "confidence": 0.80,
            },
        ))

        # 5 - Good faithfulness but low relevance.
        conditions.append((
            "grounded_but_off_question",
            faith is not None and faith > 0.8 and ar is not None and ar < 0.5,
            {
                "failed_layer": "prompt",
                "root_cause": (
                    "The answer is well-grounded in retrieved context but does "
                    "not address what the user actually asked. The LLM is "
                    "answering a related but different question."
                ),
                "suggested_fix": (
                    "Review your generation prompt. Ensure the original user "
                    "question is prominently placed. Add explicit instruction: "
                    "'Answer the specific question asked: {question}. Stay focused "
                    "on what was asked.' Consider adding a question-relevance "
                    "check before generation."
                ),
                "confidence": 0.83,
            },
        ))

        # 6 - Noise sensitive.
        conditions.append((
            "noise_sensitive",
            ns is not None and ns < 0.5 and faith is not None and faith > 0.7,
            {
                "failed_layer": "fusion_ranking",
                "root_cause": (
                    "The pipeline performs well with clean retrieval but degrades "
                    "significantly when noisy chunks are mixed with relevant ones. "
                    "The ranking or reranking step is not effectively separating "
                    "signal from noise."
                ),
                "suggested_fix": (
                    "Add or improve a reranking step. Cross-encoder rerankers "
                    "significantly improve signal-to-noise separation. Consider "
                    "increasing the gap between retrieval top_k and generation "
                    "top_k - retrieve 20, generate from top 3 after reranking. "
                    "Review RRF fusion weights if using hybrid retrieval."
                ),
                "confidence": 0.74,
            },
        ))

        # 7 - Retrieval/generation good but answer_correctness low.
        conditions.append((
            "knowledge_base_wrong",
            faith is not None and faith > 0.7
            and cp is not None and cp > 0.6
            and ac is not None and ac < 0.5,
            {
                "failed_layer": "knowledge_base",
                "root_cause": (
                    "The pipeline is working correctly but the knowledge base "
                    "contains incorrect, outdated, or incomplete information. The "
                    "model is faithfully reproducing wrong information."
                ),
                "suggested_fix": (
                    "Audit source documents for this query type. Check "
                    "publication dates - outdated clinical guidelines or "
                    "superseded regulations are a common cause of this pattern. "
                    "Add document freshness metadata and filter by recency. "
                    "Consider adding a human review step for knowledge base "
                    "entries in this domain."
                ),
                "confidence": 0.77,
            },
        ))

        available = [
            v for v in (cp, cr, faith, ar, ns, ac) if v is not None
        ]

        # 8 - All scores above threshold.
        conditions.append((
            "all_good",
            bool(available) and all(v > 0.7 for v in available) and not halluc,
            {
                "failed_layer": None,
                "root_cause": (
                    "No single clear failure layer identified. Scores indicate "
                    "reasonable pipeline quality across evaluated dimensions."
                ),
                "suggested_fix": (
                    "Expand your evaluation dataset to surface edge cases. "
                    "Consider domain-specific metrics for your use case. Run "
                    "noise_sensitivity tests to verify robustness."
                ),
                "confidence": 0.65,
            },
        ))

        matched = [(key, spec) for key, ok, spec in conditions if ok]

        evidence = {
            "context_precision": cp,
            "context_recall": cr,
            "faithfulness": faith,
            "answer_relevance": ar,
            "noise_sensitivity": ns,
            "answer_correctness": ac,
            "hallucination_detected": halluc,
        }

        if not matched:
            if not available:
                return DiagnosisResult(
                    failed_layer=None,
                    root_cause=(
                        "Insufficient metric coverage to diagnose a failing layer. "
                        "Run more metrics (at least faithfulness and context_precision)."
                    ),
                    suggested_fix=(
                        "Re-run the evaluation with the default metric set or "
                        "metrics=['faithfulness', 'context_precision', "
                        "'answer_relevance', 'hallucination']."
                    ),
                    confidence=0.4,
                    evidence=evidence,
                    secondary_issues=[],
                )
            return DiagnosisResult(
                failed_layer=None,
                root_cause=(
                    "No decisive failure pattern. Scores are mixed - some "
                    "dimensions are adequate while others are middling, but no "
                    "single layer stands out as the cause."
                ),
                suggested_fix=(
                    "Look at the lowest individual metric scores below and their "
                    "violations. Expand the evaluation dataset so weak dimensions "
                    "show a clearer signal, and add domain-specific metrics."
                ),
                confidence=0.5,
                evidence=evidence,
                secondary_issues=[],
            )

        primary_key, primary = matched[0]
        secondary = [
            self._describe(key, spec)
            for key, spec in matched[1:]
            if spec["failed_layer"] is not None
        ]

        return DiagnosisResult(
            failed_layer=primary["failed_layer"],
            root_cause=primary["root_cause"],
            suggested_fix=primary["suggested_fix"],
            confidence=primary["confidence"],
            evidence=evidence,
            secondary_issues=secondary,
        )

    @staticmethod
    def _describe(key: str, spec: Dict) -> str:
        layer = spec["failed_layer"]
        return f"[{key}] possible {layer} issue: {spec['root_cause']}"
