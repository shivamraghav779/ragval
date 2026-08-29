"""Pure-Python utilities for ragval: text, scoring, async, and JSON helpers.

Nothing in this package imports numpy, sklearn, sentence-transformers, or any
other ML dependency. Everything here is standard library only.
"""

from ragval.utils import async_utils, json_parser, scoring, text

__all__ = ["text", "scoring", "json_parser", "async_utils"]
