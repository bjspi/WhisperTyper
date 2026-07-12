"""LivePrompt trigger-word detection and stripping — pure text logic, unit-testable.

A transcription becomes a "live prompt" when one of the configured trigger words appears
within the first ``scan_depth`` words (e.g. saying "prompt, write birthday wishes …" sends
the sentence to the chat model instead of typing it out verbatim).
"""
from __future__ import annotations

from typing import List, Optional

#: Leading separators/punctuation stripped from the instruction after the trigger word.
_TRIGGER_SEPARATORS = " \t\r\n.,:;!?-–—…\"'“”‘’"


def parse_trigger_words(trigger_words_csv: str) -> List[str]:
    """Split the configured comma-separated trigger words into a clean lowercase list."""
    return [word.strip() for word in (trigger_words_csv or "").lower().split(",") if word.strip()]


def contains_trigger(text: str, trigger_words: List[str], scan_depth: int) -> bool:
    """Return True if any trigger word occurs within the first ``scan_depth`` words of ``text``."""
    if not trigger_words:
        return False
    words_to_check = text.split()[:max(0, scan_depth)]
    text_to_check = " ".join(words_to_check).lower()
    return any(trigger in text_to_check for trigger in trigger_words)


def strip_trigger(text: str, trigger_words: List[str]) -> str:
    """Return the instruction after the earliest trigger-word occurrence.

    Removes everything up to and including the first trigger occurrence, then strips leading
    separators/punctuation (e.g. "Das bitte als Anweisung. Schreib …" -> "Schreib …").
    Falls back to the full text if no trigger is found or the result would be empty.
    """
    lower = text.lower()
    best_end: Optional[int] = None
    for trigger in trigger_words:
        if not trigger:
            continue
        idx = lower.find(trigger)
        if idx != -1 and (best_end is None or idx + len(trigger) < best_end):
            best_end = idx + len(trigger)
    if best_end is None:
        return text
    remainder = text[best_end:].lstrip(_TRIGGER_SEPARATORS)
    return remainder or text
