"""Shared fixtures and a configurable MockProvider for the ragval test suite."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

import pytest

_NUMBERED_RE = re.compile(r"^\s*\d+\.\s+(.*)$", re.MULTILINE)
_CHUNK_RE = re.compile(r"\[chunk \d+\]")
_TURN_RE = re.compile(r"^\[\d+\]\s+(user|assistant):", re.MULTILINE)


def _numbered_items(prompt: str) -> List[str]:
    return [m.group(1).strip() for m in _NUMBERED_RE.finditer(prompt)]


def _chunk_count(prompt: str) -> int:
    return len(_CHUNK_RE.findall(prompt))


class MockProvider:
    """Stand-in for LiteLLMProvider.

    Behaviour, in priority order:
      1. If ``raise_next`` is set, raise it once (for retry/error tests).
      2. If a registered marker matches the prompt, return its response
         (a dict, or a callable ``prompt -> dict``).
      3. Otherwise fall back to a "healthy pipeline" default that inspects the
         prompt structure so that count-sensitive metrics get well-formed data.
    """

    def __init__(self, model_string: str = "mock/model") -> None:
        self.model_string = model_string
        self.calls: List[str] = []
        self.responses: Dict[str, Any] = {}
        self.raise_next: Optional[Exception] = None
        self.default_handler: Optional[Callable[[str], Dict[str, Any]]] = None

    # -- configuration -------------------------------------------------

    def set_response(self, marker: str, value: Any) -> "MockProvider":
        self.responses[marker] = value
        return self

    def set_default(self, handler: Callable[[str], Dict[str, Any]]) -> "MockProvider":
        self.default_handler = handler
        return self

    # -- provider interface -----------------------------------------

    async def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self.raise_next is not None:
            exc, self.raise_next = self.raise_next, None
            raise exc
        return json.dumps(self._route(prompt))

    async def complete_json(self, prompt: str) -> Dict[str, Any]:
        text = await self.complete(prompt)
        return json.loads(text)

    def complete_sync(self, prompt: str) -> str:
        from ragval.utils.async_utils import run_sync

        return run_sync(self.complete(prompt))

    def complete_json_sync(self, prompt: str) -> Dict[str, Any]:
        from ragval.utils.async_utils import run_sync

        return run_sync(self.complete_json(prompt))

    # -- routing ---------------------------------------------------

    def _route(self, prompt: str) -> Dict[str, Any]:
        for marker, value in self.responses.items():
            if marker in prompt:
                return value(prompt) if callable(value) else value
        if self.default_handler is not None:
            return self.default_handler(prompt)
        return self._default(prompt)

    @staticmethod
    def _default(prompt: str) -> Dict[str, Any]:
        p = prompt

        # ---- retrieval ----
        if "evaluating a retrieval system" in p:
            n = _chunk_count(p)
            return {
                "verdicts": [
                    {"index": i, "relevant": True, "reason": "relevant"} for i in range(n)
                ]
            }
        if "atomic, standalone\nfactual statements" in p or "atomic, standalone" in p:
            return {"statements": ["statement one", "statement two"]}
        if "checking whether retrieved context covers" in p:
            items = _numbered_items(p)
            return {
                "attributions": [
                    {"statement": s, "attributed": True, "source_chunk_index": 0}
                    for s in items
                ]
            }
        if "Rate how relevant this single retrieved chunk" in p:
            return {"score": 3, "reason": "directly answers the question"}
        if "Extract every named entity" in p:
            return {"entities": [{"text": "Metformin", "type": "drug"}]}

        # ---- generation ----
        if "Break the following answer into a list of atomic claims" in p:
            return {"claims": ["claim one", "claim two"]}
        if "checking whether each claim is supported" in p:
            items = _numbered_items(p)
            return {
                "verdicts": [
                    {"claim": c, "supported": True, "reason": "in context",
                     "supporting_chunk_index": 0}
                    for c in items
                ]
            }
        if "Generate 3 different questions" in p:
            return {"questions": ["what is X", "what is X exactly", "tell me about X"]}
        if "Break the following text into a list of atomic factual claims" in p:
            return {"claims": ["claim one", "claim two"]}
        if "semantically equivalent to" in p:
            # Only the claims listed under "Answer claims:" get a verdict.
            tail = p.split("Answer claims:", 1)[-1]
            items = _numbered_items(tail)
            return {"matches": [{"answer_claim": c, "matched": True} for c in items]}
        if "Extract every specific, checkable factual detail" in p:
            return {"entities": [{"text": "500mg", "type": "dose"}]}
        if "For each factual detail from an answer, check it against the context" in p:
            items = _numbered_items(p)
            return {
                "checks": [
                    {"entity": e, "found": True, "contradicted": False,
                     "contradiction_detail": None}
                    for e in items
                ]
            }
        if "List the key topics" in p:
            return {"key_topics": ["topic one", "topic two"]}
        if "for each key topic, whether the summary addresses it" in p:
            items = _numbered_items(p)
            return {"coverage": [{"topic": t, "addressed": True} for t in items]}

        # ---- ranking labels ----
        if "judge each retrieved chunk as relevant (true)" in p:
            n = _chunk_count(p)
            return {"labels": [{"index": i, "relevant": i == 0} for i in range(n)]}
        if "assign each retrieved chunk a graded relevance" in p:
            n = _chunk_count(p)
            return {
                "grades": [
                    {"index": i, "grade": 3 if i == 0 else 1, "reason": "x"}
                    for i in range(n)
                ]
            }

        # ---- safety ----
        if "Evaluate the answer for bias" in p:
            return {
                "bias_detected": False,
                "bias_types": [],
                "overall_bias_score": 0.0,
                "reasoning": "no bias",
            }
        if "Evaluate the answer for toxic content" in p:
            return {
                "toxicity_detected": False,
                "toxicity_types": [],
                "toxicity_score": 0.0,
                "context_appropriate": True,
                "reasoning": "benign",
            }
        if "stays within the expected domain" in p:
            return {
                "on_topic": True,
                "off_topic_content": [],
                "adherence_score": 1.0,
                "reasoning": "on topic",
            }

        # ---- agentic ----
        if "compare the actual arguments against the expected arguments" in p:
            return {"tool_argument_verdicts": [{"tool": "search", "correct": True, "issues": []}]}
        if "Evaluate whether an agent completed the user's task" in p:
            return {
                "task_completed": True,
                "completion_score": 0.95,
                "unmet_requirements": [],
                "reasoning": "done",
            }
        if "estimate the minimum number of steps" in p:
            return {"min_steps_estimate": 2, "redundant_steps": [], "efficiency_score": 0.9}
        if "Compare an agent's declared plan against what it actually did" in p:
            return {"adherence_score": 0.9, "plan_deviations": [], "reasoning": "followed"}
        if "Evaluate the quality of an agent's plan" in p:
            return {
                "quality_score": 0.9,
                "strengths": ["clear"],
                "weaknesses": [],
                "reasoning": "good plan",
            }
        if "Compare an agent's final answer to the expected outcome" in p:
            return {
                "goal_achieved": True,
                "accuracy_score": 0.9,
                "gaps": [],
                "reasoning": "matches",
            }

        # ---- conversation ----
        if "Extract the key facts that the user established" in p:
            return {"established_facts": ["user is allergic to penicillin", "user is 45"]}
        if "whether each fact is referenced" in p:
            items = _numbered_items(p)
            return {
                "retention_checks": [
                    {"fact": f, "referenced_in_later_turn": True, "correctly_used": True}
                    for f in items
                ]
            }
        if "decide whether the user's need\nwas satisfied" in p or "the user's need" in p:
            turns = _TURN_RE.findall(p)
            n_user = max(1, turns.count("user"))
            return {
                "need_satisfaction": [
                    {"user_turn": f"turn {i}", "satisfied": True,
                     "partially_satisfied": False, "gaps": []}
                    for i in range(n_user)
                ]
            }
        if "Evaluate each assistant turn below" in p:
            idxs = [int(m) for m in re.findall(r"\[turn (\d+)\]", p)]
            return {
                "adherence_verdicts": [
                    {"turn_index": i, "adheres": True, "violation_type": None}
                    for i in idxs
                ],
                "overall_adherence": 1.0,
            }

        # ---- clinical ----
        if "Extract every drug\nname mentioned" in p or "Extract every drug" in p:
            return {"drug_names": ["Metformin"]}
        if "For each drug name from an answer" in p:
            items = _numbered_items(p)
            return {"verdicts": [{"drug": d, "verified": True} for d in items]}
        if "For each dosing statement extracted from an answer" in p:
            items = _numbered_items(p)
            return {
                "verdicts": [{"dosing": d, "verified": True, "note": "ok"} for d in items]
            }
        if "extract every contraindication" in p:
            return {"contraindications": ["renal impairment"]}
        if "mentions or accounts for it" in p:
            items = _numbered_items(p)
            return {"coverage": [{"contraindication": c, "mentioned": True} for c in items]}

        # ---- legal ----
        if "implies or requires a specific jurisdiction" in p:
            return {
                "question_implies_jurisdiction": True,
                "answer_specifies_jurisdiction": "clearly",
                "reasoning": "states California",
            }
        if "Extract every legal\ncitation" in p or "Extract every legal" in p:
            return {"citations": []}
        if "statutes and authorities cited" in p:
            return {"currency_score": 1.0, "potentially_outdated": [], "reasoning": "current"}

        # ---- financial ----
        if "Extract every specific\nfinancial figure" in p or "Extract every specific" in p:
            return {"numbers": []}
        if "question involves financial products" in p:
            return {
                "requires_regulatory_context": False,
                "regulatory_context_present": False,
                "regulators_mentioned": [],
                "reasoning": "general query",
            }

        # ---- extended retrieval ----
        if "how much each retrieved chunk was actually USED" in p:
            n = _chunk_count(p)
            return {
                "verdicts": [
                    {"index": i, "used": i == 0, "reason": "x"} for i in range(n)
                ]
            }
        if "contains enough information to fully" in p:
            return {
                "sufficient": True,
                "missing_information": [],
                "sufficiency_score": 0.9,
                "reasoning": "covers the question",
            }

        # ---- extended generation ----
        if "Break the question into its distinct information needs" in p:
            return {
                "requirements": [
                    {"need": "part one", "addressed": True},
                    {"need": "part two", "addressed": True},
                ],
                "reasoning": "both parts answered",
            }
        if "Rate the coherence of the answer" in p:
            return {"score": 5, "issues": [], "reasoning": "well structured"}
        if "Rate the fluency of the answer" in p:
            return {"score": 5, "issues": [], "reasoning": "polished"}
        if "Rate how concise the answer is" in p:
            return {"conciseness_score": 0.9, "redundant_spans": [], "reasoning": "tight"}
        if "either answered the question or declined" in p:
            return {
                "did_refuse": False,
                "context_supports_answer": True,
                "appropriate": True,
                "reasoning": "context supports the answer",
            }
        if "quoted phrases, figures, or references attributed" in p:
            return {"citations": []}

        # ---- extended safety ----
        if "personally identifiable information (PII)" in p:
            return {
                "pii_detected": False,
                "pii_items": [],
                "leakage_score": 0.0,
                "reasoning": "no PII",
            }
        if "professionalism and appropriateness of the answer's tone" in p:
            return {"score": 1.0, "tone_issues": [], "reasoning": "professional"}

        # ---- extended conversation ----
        if "For each assistant turn, decide whether it is relevant" in p:
            idxs = [
                i for i, ln in enumerate(p.splitlines())
                if ln.strip().startswith("[") and "assistant:" in ln
            ]
            n = len(idxs) or 1
            return {
                "turn_relevancy": [
                    {"turn_index": i, "relevant": True, "note": None} for i in range(n)
                ],
                "overall_relevancy": 1.0,
            }

        # ---- custom ----
        if "ordered list of concrete evaluation steps" in p:
            return {"evaluation_steps": ["check the criteria", "score it"]}
        if "Apply the criteria and evaluation steps" in p:
            return {"reasoning": "meets criteria", "score": 0.9, "violations": []}
        if "selecting the single rubric level" in p:
            return {"selected_level": 5, "reasoning": "excellent", "specific_issues": []}
        if "binary pass/fail verdict" in p:
            return {"passed": True, "score": 1.0, "reason": "passes"}

        return {"score": 1.0, "reasoning": "default mock response"}


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_provider() -> MockProvider:
    return MockProvider()


@pytest.fixture
def sample_question() -> str:
    return "What is the first-line treatment for Type 2 diabetes?"


@pytest.fixture
def sample_answer() -> str:
    return (
        "Metformin is the recommended first-line medication for Type 2 diabetes "
        "per ADA guidelines with dose 500mg twice daily."
    )


@pytest.fixture
def sample_contexts() -> List[str]:
    return [
        "ADA guidelines recommend metformin as the first-line pharmacologic "
        "agent for Type 2 diabetes. Typical starting dose is 500mg twice daily.",
        "Type 2 diabetes management combines lifestyle modification with "
        "pharmacotherapy. Metformin is preferred unless contraindicated.",
        "The mitochondria is the powerhouse of the cell and produces ATP.",
    ]


@pytest.fixture
def sample_ground_truth() -> str:
    return (
        "Metformin is the first-line pharmacologic treatment for Type 2 diabetes "
        "according to ADA guidelines, usually started at 500mg twice daily."
    )


@pytest.fixture
def sample_tool_calls() -> List[Dict[str, Any]]:
    return [
        {"name": "search_guidelines", "arguments": {"query": "type 2 diabetes first line"}},
        {"name": "lookup_drug", "arguments": {"drug": "metformin"}},
    ]


@pytest.fixture
def sample_turns() -> List[Dict[str, Any]]:
    return [
        {"role": "user", "content": "I'm 45 and allergic to penicillin. I have a sore throat."},
        {"role": "assistant", "content": "Noted: age 45, penicillin allergy. How long have you had the sore throat?"},
        {"role": "user", "content": "About three days, with a mild fever."},
        {"role": "assistant", "content": "Given your penicillin allergy, if antibiotics are needed a macrolide like azithromycin is an option. Please see a clinician for a strep test."},
        {"role": "user", "content": "Can I take amoxicillin my friend gave me?"},
        {"role": "assistant", "content": "No - amoxicillin is a penicillin and you are allergic. Do not take it."},
    ]


@pytest.fixture
def make_evaluator(mock_provider):
    """Build a RAGEvaluator whose provider is the MockProvider."""
    from ragval.evaluator import RAGEvaluator

    def _make(**kwargs) -> RAGEvaluator:
        ev = RAGEvaluator(model="mock/model", **kwargs)
        ev.provider = mock_provider
        return ev

    return _make
