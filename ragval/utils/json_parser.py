"""Best-effort JSON extraction from LLM responses.

LLMs return JSON wrapped in prose, fenced code blocks, or with trailing commas
despite instructions. ``extract_json`` tries progressively more lenient
strategies before giving up.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from ragval.exceptions import MetricComputationError

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _loads(candidate: str) -> Any:
    return json.loads(candidate)


def _try_repair(candidate: str) -> Any:
    """Remove trailing commas, then parse."""
    repaired = re.sub(r",(\s*[}\]])", r"\1", candidate)
    return json.loads(repaired)


def extract_json(text: str) -> Dict[str, Any]:
    """Return a dict parsed from ``text``.

    Strategy order:
      1. ``json.loads`` on the stripped text.
      2. Strip ```` ```json ... ``` ```` fences, then parse.
      3. Slice from the first ``{`` to the last ``}``, then parse.
      4. Slice from the first ``[`` to the last ``]``, then parse (wrapped as
         ``{"items": [...]}`` when the top-level value is a list).
      5. Each of the above again with a trailing-comma repair pass.

    Raises :class:`MetricComputationError` (with the raw text attached) on
    total failure.
    """
    if text is None:
        raise MetricComputationError(
            "LLM returned no text to parse", reason="empty response", raw_response=""
        )

    stripped = text.strip()

    # Try 1: direct parse.
    for parser in (_loads, _try_repair):
        try:
            result = parser(stripped)
            return _as_dict(result)
        except (json.JSONDecodeError, ValueError):
            pass

    # Try 2: fenced code block.
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        inner = fence_match.group(1).strip()
        for parser in (_loads, _try_repair):
            try:
                return _as_dict(parser(inner))
            except (json.JSONDecodeError, ValueError):
                pass

    # Try 3: first { .. last }.
    if "{" in stripped and "}" in stripped:
        candidate = stripped[stripped.index("{") : stripped.rindex("}") + 1]
        for parser in (_loads, _try_repair):
            try:
                return _as_dict(parser(candidate))
            except (json.JSONDecodeError, ValueError):
                pass

    # Try 4: first [ .. last ].
    if "[" in stripped and "]" in stripped:
        candidate = stripped[stripped.index("[") : stripped.rindex("]") + 1]
        for parser in (_loads, _try_repair):
            try:
                return _as_dict(parser(candidate))
            except (json.JSONDecodeError, ValueError):
                pass

    raise MetricComputationError(
        "Could not extract valid JSON from the LLM response",
        reason="all parse strategies failed",
        raw_response=text,
    )


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {"items": value}
    raise ValueError(f"parsed JSON is a {type(value).__name__}, not an object")
