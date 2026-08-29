"""Provider package. One integration: LiteLLM."""

from __future__ import annotations

from typing import Any

from ragval.providers.litellm_provider import LiteLLMProvider

__all__ = ["LiteLLMProvider", "get_provider"]


def get_provider(model_string: str, **kwargs: Any) -> LiteLLMProvider:
    """Construct a :class:`LiteLLMProvider` for ``model_string``."""
    return LiteLLMProvider(model_string, **kwargs)
