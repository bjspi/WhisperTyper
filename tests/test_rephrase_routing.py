"""Which post-processing a finished transcription gets, by priority."""
from __future__ import annotations

from copy import deepcopy

import pytest

from app.core.constants import DEFAULT_CONFIG
from app.core.rephrase_routing import RephrasePlan, plan_rephrase, usable_transcript


def config(**overrides):
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg.update(liveprompt_enabled=False, generic_rephrase_enabled=False, liveprompt_trigger_words="prompt",
               liveprompt_trigger_word_scan_depth=3, liveprompt_system_prompt="LIVE", liveprompt_strip_trigger=False,
               rephrase_use_selection_context=False, generic_rephrase_prompt="Polish")
    cfg.update(overrides)
    return cfg


@pytest.mark.parametrize(("text", "expected"), [
    ('"Hello there."', "Hello there."),
    ("  “Quoted”  ", "Quoted"),
    ("", None),
    ("The transcription prompt", None),  # The model echoed its prompt: nothing was said.
])
def test_usable_transcript_strips_wrappers_and_rejects_echoed_prompt(text, expected):
    assert usable_transcript(text, " the transcription PROMPT ") == expected


def test_palette_prompt_overrides_liveprompt_and_generic_rephrasing():
    cfg = config(liveprompt_enabled=True, generic_rephrase_enabled=True)
    assert plan_rephrase("prompt write a poem", cfg, "  Translate  ", "selection") == RephrasePlan(
        "Translate", "prompt write a poem")


@pytest.mark.parametrize(("strip", "use_context", "expected"), [
    (False, False, RephrasePlan("LIVE", "Okay prompt, write a poem", "")),
    (True, True, RephrasePlan("LIVE", "write a poem", "selection")),
])
def test_liveprompt_trigger_within_scan_depth(strip, use_context, expected):
    cfg = config(liveprompt_enabled=True, generic_rephrase_enabled=True,
                 liveprompt_strip_trigger=strip, rephrase_use_selection_context=use_context)
    assert plan_rephrase("Okay prompt, write a poem", cfg, None, "selection") == expected


def test_trigger_beyond_scan_depth_falls_through_to_generic_rephrasing():
    cfg = config(liveprompt_enabled=True, generic_rephrase_enabled=True)
    plan = plan_rephrase("one two three prompt four", cfg, None, "selection")
    assert plan == RephrasePlan("", "Polish\n\nText: one two three prompt four")


def test_without_any_rephrasing_the_text_is_delivered_unchanged():
    assert plan_rephrase("prompt write a poem", config(), "   ", "selection") is None
