"""Decide how a finished transcription is post-processed — pure logic, no Qt, no I/O.

Priority: an explicit recording-palette prompt, then a LivePrompt trigger word, then the
generic rephrase prompt; otherwise the transcription is delivered as-is.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from app.core import liveprompt

#: Quote characters and spaces a model may wrap around the transcript.
_TRANSCRIPT_WRAPPERS = '"\'“”‘’ '


@dataclass(frozen=True)
class RephrasePlan:
    """One chat-completion request that replaces the raw transcription."""

    system_prompt: str
    user_prompt: str
    context: str = ""


def usable_transcript(text: str, transcription_prompt: str) -> Optional[str]:
    """Strip wrapping quotes; None when nothing was said or the model echoed its prompt."""
    processed = text.strip(_TRANSCRIPT_WRAPPERS)
    prompt = transcription_prompt.strip()
    if not processed or (prompt and processed.lower() == prompt.lower()):
        return None
    return processed


def plan_rephrase(text: str, config: Mapping[str, Any], transformation_prompt: Optional[str],
                  selection_context: str) -> Optional[RephrasePlan]:
    """Return the rephrasing request for ``text``, or None to deliver it unchanged."""
    # 1. An explicit recording-palette choice overrides every automatic rephrasing mode.
    if transformation_prompt and transformation_prompt.strip():
        return RephrasePlan(system_prompt=transformation_prompt.strip(), user_prompt=text)

    # 2. LivePrompting via trigger words: the transcription itself is the instruction.
    if config["liveprompt_enabled"]:
        trigger_words = liveprompt.parse_trigger_words(config.get("liveprompt_trigger_words", ""))
        scan_depth = config.get("liveprompt_trigger_word_scan_depth", 5)
        if liveprompt.contains_trigger(text, trigger_words, scan_depth):
            # Optionally drop the trigger word and everything before it, so only the
            # actual instruction after it is sent to the model.
            instruction = text
            if config.get("liveprompt_strip_trigger", False):
                instruction = liveprompt.strip_trigger(text, trigger_words)
            context = selection_context if config["rephrase_use_selection_context"] else ""
            return RephrasePlan(config["liveprompt_system_prompt"], instruction, context)

    # 3. Generic rephrasing: combine the generic prompt with the transcription.
    if config["generic_rephrase_enabled"]:
        return RephrasePlan(system_prompt="", user_prompt=f"{config['generic_rephrase_prompt']}\n\nText: {text}")

    return None
