"""Decide how a finished transcription is post-processed — pure logic, no Qt, no I/O.

Priority: an explicit recording-palette choice, then a LivePrompt trigger word (the active
instruction entry), then the automatically applied prompt; otherwise the transcription is
delivered as-is.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from app.core import liveprompt
from app.core.api_keys import rephrasing_configured
from app.core.prompts import instruction_entry

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
                  selection_context: str, auto_prompt: Optional[str] = None,
                  instruction_selected: bool = False) -> Optional[RephrasePlan]:
    """Return the rephrasing request for ``text``, or None to deliver it unchanged.

    ``instruction_selected`` marks a palette click on the instruction entry: the dictation is
    then carried out as an order, with the selection context like a trigger word would add.
    """
    instruction = instruction_entry(config.get("post_rephrasing_entries", []))
    context = selection_context if instruction["use_selection_context"] else ""

    # 1. An explicit recording-palette choice overrides every automatic rephrasing mode.
    if transformation_prompt and transformation_prompt.strip():
        return RephrasePlan(system_prompt=transformation_prompt.strip(), user_prompt=text,
                            context=context if instruction_selected else "")

    # 2. LivePrompting via trigger words: the transcription itself is the instruction.
    if instruction["enabled"] and rephrasing_configured(config):
        trigger_words = liveprompt.parse_trigger_words(instruction["trigger_words"])
        if liveprompt.contains_trigger(text, trigger_words, instruction["scan_depth"]):
            # Optionally drop the trigger word and everything before it, so only the
            # actual instruction after it is sent to the model.
            order = liveprompt.strip_trigger(text, trigger_words) if instruction["strip_trigger"] else text
            return RephrasePlan(instruction["text"], order, context)

    # 3. The prompt marked "apply automatically" in the Prompts tab.
    if auto_prompt and auto_prompt.strip():
        return RephrasePlan(system_prompt=auto_prompt.strip(), user_prompt=text)

    return None
