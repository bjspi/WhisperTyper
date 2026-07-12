"""Lightweight text utilities: token estimation, mojibake repair, small string helpers."""

from __future__ import annotations

import re
from typing import Any

import ftfy

TOKEN_PATTERN = re.compile(r"\w+|[^\s\w]")  # crude approximation (words or single punctuation)
_MODEL_ANNOTATION_PATTERN = re.compile(r"\s*\(.*?\)")  # e.g. "whisper-1 (openai)" -> "whisper-1"


def demojibake(value: Any) -> Any:
    """Repair UTF-8 text that was mis-decoded as a single-byte codepage, via ``ftfy``.

    Covers both the latin-1 case ('IntelÂ®' -> 'Intel®') and the Windows-1252 case
    ('Ãœbersetzer' -> 'Übersetzer'), plus nested/multi-round mojibake, which ftfy detects
    automatically. Uses ``fix_encoding`` (encoding repair only) rather than ``fix_text`` so it
    never touches quotes/whitespace in user prompts. Only strings carrying the tell-tale markers
    are touched (so clean text is never re-encoded); non-strings pass through unchanged.
    """
    if not isinstance(value, str) or not any(marker in value for marker in ("Ã", "Â")):
        return value
    return ftfy.fix_encoding(value)


def estimate_tokens(text: str) -> int:
    """Very lightweight approximate tokenizer.

    NOTE: This is NOT identical to Whisper's exact tokenizer but good enough for length validation.
    """
    if not text:
        return 0
    return len(TOKEN_PATTERN.findall(text))


def clean_model_name(model: str) -> str:
    """Strip the display-only provider annotation from a model option ("whisper-1 (openai)")."""
    return _MODEL_ANNOTATION_PATTERN.sub("", model or "").strip()


def shorten(text: str, limit: int = 100) -> str:
    """Collapse whitespace and clip to ``limit`` chars (+ "[…]") for compact previews."""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + " […]"
