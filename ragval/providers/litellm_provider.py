"""The single provider integration for ragval.

Every LLM call in the library goes through :class:`LiteLLMProvider`. Metric
files must never ``import litellm`` directly. LiteLLM gives us OpenAI,
Anthropic, Groq, Azure, Ollama, and 100+ providers behind one model string.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Dict, Optional

from ragval.exceptions import MetricComputationError, ProviderError
from ragval.utils import async_utils, json_parser

# "try again in 12.5s", "retry after 30 seconds", "Retry-After: 8"
_RETRY_AFTER_RE = re.compile(
    r"(?:retry[- ]after|try again in|in)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(m|min|minute|s|sec|second)?",
    re.IGNORECASE,
)

# Map a model-string prefix to the environment variable that holds its key.
_ENV_KEY_BY_PREFIX: Dict[str, str] = {
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "azure": "AZURE_API_KEY",
    "cohere": "COHERE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "together_ai": "TOGETHERAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

_JSON_NUDGE = (
    "\n\nRespond with valid JSON only. No markdown fences. "
    "No explanation. Just JSON."
)
_JSON_STRICTER = (
    "\n\nYour previous response was not valid JSON. Return ONLY a single JSON "
    "object. Start your response with '{' and end it with '}'. Do not write "
    "anything else."
)


class LiteLLMProvider:
    """Async-first wrapper around ``litellm.acompletion``."""

    def __init__(
        self,
        model_string: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: int = 30,
        max_concurrency: int = 8,
    ) -> None:
        if not model_string or not isinstance(model_string, str):
            raise ProviderError(
                "model_string must be a non-empty LiteLLM model string, e.g. "
                "'groq/llama-3.3-70b-versatile'",
                model=model_string,
            )
        self.model_string = model_string
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.api_key = api_key or self._resolve_api_key(model_string)
        self._max_retries = 3
        # Cap simultaneous in-flight calls. ``evaluate(metrics="all")`` fans out
        # ~40 metric calls at once; without this they thunder-herd a
        # rate-limited provider and most fail together. Lower it (e.g. 2) for
        # strict free tiers.
        self.max_concurrency = max(1, max_concurrency)
        self._semaphore: Optional[asyncio.Semaphore] = None
        # Rate limits get their own, more patient budget: free-tier windows are
        # measured in tens of seconds, far longer than exponential backoff of 3.
        self._max_rate_limit_retries = 6
        self._max_backoff_seconds = 60.0

    # -- construction helpers -------------------------------------------------

    @staticmethod
    def _prefix(model_string: str) -> str:
        return model_string.split("/", 1)[0].lower() if "/" in model_string else ""

    def _resolve_api_key(self, model_string: str) -> Optional[str]:
        prefix = self._prefix(model_string)
        env_var = _ENV_KEY_BY_PREFIX.get(prefix)
        if env_var:
            return os.environ.get(env_var)
        # Fall back to common vars; LiteLLM will also read env vars itself.
        return os.environ.get("OPENAI_API_KEY")

    # -- low level call -----------------------------------------------------

    async def _acompletion(self, prompt: str) -> str:
        try:
            import litellm  # imported lazily so import errors are actionable
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "litellm is not installed. Run: pip install litellm",
                model=self.model_string,
            ) from exc

        kwargs: Dict[str, Any] = {
            "model": self.model_string,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base

        response = await litellm.acompletion(**kwargs)
        try:
            return response["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                "LLM response had an unexpected shape",
                model=self.model_string,
                provider_message=str(response)[:300],
            ) from exc

    def _classify(self, exc: Exception) -> str:
        """Return one of 'auth', 'rate_limit', 'timeout', 'other'."""
        name = type(exc).__name__.lower()
        text = str(exc).lower()
        if "auth" in name or "authenticationerror" in name or "invalid api key" in text:
            return "auth"
        if "ratelimit" in name or "rate limit" in text or "429" in text:
            return "rate_limit"
        if "timeout" in name or isinstance(exc, asyncio.TimeoutError):
            return "timeout"
        return "other"

    def _retry_after_seconds(self, exc: Exception) -> Optional[float]:
        """Best-effort parse of a provider-suggested wait from the error text."""
        match = _RETRY_AFTER_RE.search(str(exc))
        if not match:
            return None
        value = float(match.group(1))
        unit = (match.group(2) or "s").lower()
        if unit.startswith("m"):
            value *= 60.0
        return min(value + 1.0, self._max_backoff_seconds)

    def _get_semaphore(self) -> asyncio.Semaphore:
        # Created lazily so it binds to the running loop, not import-time.
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore

    async def complete(self, prompt: str) -> str:
        """Return raw text from the model, retrying transient failures.

        Rate limits are retried up to ``_max_rate_limit_retries`` times, honoring
        any provider-suggested ``Retry-After`` wait; other transient errors get
        ``_max_retries`` attempts with exponential backoff. At most
        ``max_concurrency`` calls run at once.
        """
        async with self._get_semaphore():
            return await self._complete_inner(prompt)

    async def _complete_inner(self, prompt: str) -> str:
        last_exc: Optional[Exception] = None
        timeout_retried = False
        general_attempts = 0
        rate_limit_attempts = 0

        while True:
            try:
                return await self._acompletion(prompt)
            except Exception as exc:  # noqa: BLE001 - we re-raise as ProviderError
                last_exc = exc
                kind = self._classify(exc)

                if kind == "auth":
                    raise ProviderError(
                        "Authentication failed. Check the API key for "
                        f"{self._prefix(self.model_string) or 'your provider'}.",
                        model=self.model_string,
                        provider_message=str(exc),
                        retries=general_attempts + rate_limit_attempts,
                    ) from exc

                if kind == "timeout":
                    if timeout_retried:
                        raise ProviderError(
                            "The request timed out twice. Try a faster model or "
                            "raise the provider timeout.",
                            model=self.model_string,
                            provider_message=str(exc),
                            retries=general_attempts + rate_limit_attempts + 1,
                        ) from exc
                    timeout_retried = True
                    continue  # retry immediately once

                if kind == "rate_limit":
                    rate_limit_attempts += 1
                    if rate_limit_attempts > self._max_rate_limit_retries:
                        break
                    wait = self._retry_after_seconds(exc)
                    if wait is None:
                        wait = min(2 ** rate_limit_attempts, self._max_backoff_seconds)
                    await asyncio.sleep(wait)
                    continue

                # other: exponential backoff then retry
                general_attempts += 1
                if general_attempts >= self._max_retries:
                    break
                await asyncio.sleep(min(2 ** general_attempts, self._max_backoff_seconds))

        raise ProviderError(
            "The LLM call failed after all retries.",
            model=self.model_string,
            provider_message=str(last_exc) if last_exc else None,
            retries=general_attempts + rate_limit_attempts,
        )

    async def complete_json(self, prompt: str) -> Dict[str, Any]:
        """Return a parsed JSON object from the model."""
        first = await self.complete(prompt + _JSON_NUDGE)
        try:
            return json_parser.extract_json(first)
        except MetricComputationError:
            pass

        second = await self.complete(prompt + _JSON_STRICTER)
        try:
            return json_parser.extract_json(second)
        except MetricComputationError as exc:
            raise MetricComputationError(
                "The model did not return valid JSON after two attempts.",
                reason="json parse failed twice",
                raw_response=second,
            ) from exc

    # -- sync wrappers -----------------------------------------------------

    def complete_sync(self, prompt: str) -> str:
        return async_utils.run_sync(self.complete(prompt))

    def complete_json_sync(self, prompt: str) -> Dict[str, Any]:
        return async_utils.run_sync(self.complete_json(prompt))
