"""Exception hierarchy for ragval.

Every exception carries a human-readable message that tells the user exactly
what went wrong and what to do about it. Metrics never propagate these — the
evaluator and metric base classes catch them and convert to error results —
but they are raised freely inside providers and utilities.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class RAGEvalError(Exception):
    """Base class for all ragval errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: Dict[str, Any] = details or {}

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.details:
            extra = ", ".join(f"{k}={v!r}" for k, v in self.details.items())
            return f"{self.message} ({extra})"
        return self.message


class ProviderError(RAGEvalError):
    """Raised when the LLM provider fails after all retries."""

    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        provider_message: Optional[str] = None,
        retries: int = 0,
    ) -> None:
        super().__init__(
            message,
            {"model": model, "provider_message": provider_message, "retries": retries},
        )
        self.model = model
        self.provider_message = provider_message
        self.retries = retries

    def __str__(self) -> str:
        parts = [self.message]
        if self.model:
            parts.append(f"model={self.model!r}")
        if self.provider_message:
            parts.append(f"provider said: {self.provider_message}")
        if self.retries:
            parts.append(f"after {self.retries} retries")
        parts.append(
            "Check your API key env var, the model string, and network access."
        )
        return " | ".join(parts)


class MetricComputationError(RAGEvalError):
    """Raised when a metric cannot compute a result (e.g. unparseable LLM JSON)."""

    def __init__(
        self,
        message: str,
        metric_name: Optional[str] = None,
        reason: Optional[str] = None,
        raw_response: Optional[str] = None,
    ) -> None:
        super().__init__(
            message,
            {
                "metric_name": metric_name,
                "reason": reason,
                "raw_response": raw_response,
            },
        )
        self.metric_name = metric_name
        self.reason = reason
        self.raw_response = raw_response

    def __str__(self) -> str:
        parts = [self.message]
        if self.metric_name:
            parts.append(f"metric={self.metric_name!r}")
        if self.reason:
            parts.append(f"reason: {self.reason}")
        if self.raw_response:
            snippet = self.raw_response[:500]
            parts.append(f"raw response: {snippet}")
        return " | ".join(parts)


class DomainNotFoundError(RAGEvalError):
    """Raised when a requested domain or metric name is not registered."""

    def __init__(self, requested: str, available: Optional[List[str]] = None) -> None:
        available = available or []
        message = (
            f"{requested!r} is not registered. "
            f"Available: {', '.join(sorted(available)) if available else '(none)'}"
        )
        super().__init__(message, {"requested": requested, "available": available})
        self.requested = requested
        self.available = available


class GroundTruthRequiredError(RAGEvalError):
    """Raised when a metric that needs a reference answer is called without one."""

    def __init__(self, metric_name: str) -> None:
        message = (
            f"Metric {metric_name!r} requires a ground_truth reference answer. "
            f"Pass ground_truth=... to evaluate()."
        )
        super().__init__(message, {"metric_name": metric_name})
        self.metric_name = metric_name


class InvalidInputError(RAGEvalError):
    """Raised when user-supplied input is malformed."""

    def __init__(self, field: str, reason: str) -> None:
        message = f"Invalid input for {field!r}: {reason}"
        super().__init__(message, {"field": field, "reason": reason})
        self.field = field
        self.reason = reason
