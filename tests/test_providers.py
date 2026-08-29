"""Provider routing, retries, and JSON extraction."""

from __future__ import annotations

import pytest

from ragval.exceptions import MetricComputationError, ProviderError
from ragval.providers import get_provider
from ragval.providers.litellm_provider import LiteLLMProvider
from ragval.utils import json_parser


def test_model_prefix_routes_api_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "groq-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    groq = LiteLLMProvider("groq/llama-3.3-70b-versatile")
    openai = LiteLLMProvider("openai/gpt-4o-mini")
    assert groq.api_key == "groq-secret"
    assert openai.api_key == "openai-secret"


def test_explicit_api_key_wins(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "env-key")
    p = LiteLLMProvider("groq/llama-3.3-70b-versatile", api_key="explicit")
    assert p.api_key == "explicit"


def test_empty_model_string_raises():
    with pytest.raises(ProviderError):
        LiteLLMProvider("")


@pytest.mark.asyncio
async def test_retry_on_rate_limit(monkeypatch):
    p = LiteLLMProvider("groq/llama-3.3-70b-versatile", api_key="k")
    calls = {"n": 0}

    class RateLimitError(Exception):
        pass

    async def fake(prompt):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RateLimitError("rate limit exceeded (429)")
        return "recovered"

    monkeypatch.setattr(p, "_acompletion", fake)
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    out = await p.complete("hi")
    assert out == "recovered"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_provider_error_on_final_failure(monkeypatch):
    p = LiteLLMProvider("groq/llama-3.3-70b-versatile", api_key="k")

    async def always_fail(prompt):
        raise RuntimeError("upstream exploded")

    monkeypatch.setattr(p, "_acompletion", always_fail)
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    with pytest.raises(ProviderError):
        await p.complete("hi")


@pytest.mark.asyncio
async def test_auth_error_fails_immediately(monkeypatch):
    p = LiteLLMProvider("openai/gpt-4o-mini", api_key="k")
    calls = {"n": 0}

    async def fail_auth(prompt):
        calls["n"] += 1
        raise Exception("AuthenticationError: invalid api key")

    monkeypatch.setattr(p, "_acompletion", fail_auth)
    with pytest.raises(ProviderError):
        await p.complete("hi")
    assert calls["n"] == 1


def test_extract_json_plain():
    assert json_parser.extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_markdown_fence():
    text = "Here you go:\n```json\n{\"score\": 0.8}\n```\nthanks"
    assert json_parser.extract_json(text) == {"score": 0.8}


def test_extract_json_embedded_braces():
    text = "The result is {\"verdict\": \"PASS\"} overall."
    assert json_parser.extract_json(text) == {"verdict": "PASS"}


def test_extract_json_list_wrapped():
    assert json_parser.extract_json("[1, 2, 3]") == {"items": [1, 2, 3]}


def test_extract_json_failure_includes_raw():
    with pytest.raises(MetricComputationError) as exc:
        json_parser.extract_json("not json at all")
    assert "not json at all" in str(exc.value)


def test_get_provider_returns_litellm_provider():
    assert isinstance(get_provider("ollama/llama3"), LiteLLMProvider)


async def _no_sleep(*_a, **_k):
    return None
