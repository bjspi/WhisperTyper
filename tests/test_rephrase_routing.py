"""Which post-processing a finished transcription gets, by priority."""
from __future__ import annotations

from copy import deepcopy

import pytest

from app.core.constants import DEFAULT_CONFIG
from app.core.rephrase_routing import RephrasePlan, plan_rephrase, usable_transcript

OPENAI_KEY = {"id": "o", "name": "OpenAI", "provider": "openai", "key": "sk-test-0000000000"}


def config(enabled=False, strip=False, use_context=False, api=True):
    """Default config with a configured rephrasing API and an instruction entry with the given settings."""
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["api_key_profiles"] = [OPENAI_KEY] if api else []
    cfg["post_rephrasing_entries"] = [
        {"kind": "instruction", "caption": "Go", "text": "LIVE", "enabled": enabled, "trigger_words": "prompt",
         "scan_depth": 3, "strip_trigger": strip, "use_selection_context": use_context},
        {"caption": "Polish", "text": "Polish it"},
    ]
    return cfg


@pytest.mark.parametrize(("text", "expected"), [
    ('"Hello there."', "Hello there."),
    ("  “Quoted”  ", "Quoted"),
    ("", None),
    ("The transcription prompt", None),  # The model echoed its prompt: nothing was said.
])
def test_usable_transcript_strips_wrappers_and_rejects_echoed_prompt(text, expected):
    assert usable_transcript(text, " the transcription PROMPT ") == expected


def test_palette_prompt_overrides_liveprompt_and_the_automatic_prompt():
    cfg = config(enabled=True, use_context=True)
    assert plan_rephrase("prompt write a poem", cfg, "  Translate  ", "selection", "Polish") == RephrasePlan(
        "Translate", "prompt write a poem")


def test_palette_click_on_the_instruction_carries_out_the_dictation_with_context():
    cfg = config(enabled=True, use_context=True)
    plan = plan_rephrase("write a poem", cfg, "LIVE", "selection", "Polish", instruction_selected=True)
    assert plan == RephrasePlan("LIVE", "write a poem", "selection")


@pytest.mark.parametrize(("strip", "use_context", "expected"), [
    (False, False, RephrasePlan("LIVE", "Okay prompt, write a poem", "")),
    (True, True, RephrasePlan("LIVE", "write a poem", "selection")),
])
def test_liveprompt_trigger_within_scan_depth(strip, use_context, expected):
    cfg = config(enabled=True, strip=strip, use_context=use_context)
    # A LivePrompt trigger takes precedence over the automatic prompt.
    assert plan_rephrase("Okay prompt, write a poem", cfg, None, "selection", "Polish") == expected


@pytest.mark.parametrize("cfg", [config(enabled=False), config(enabled=True, api=False)],
                         ids=["instruction switched off", "rephrasing API not configured"])
def test_trigger_word_is_ignored_without_an_active_instruction_or_api(cfg):
    assert plan_rephrase("prompt write a poem", cfg, None, "selection", None) is None


def test_trigger_beyond_scan_depth_falls_through_to_the_automatic_prompt():
    cfg = config(enabled=True)
    plan = plan_rephrase("one two three prompt four", cfg, None, "selection", " Polish ")
    assert plan == RephrasePlan("Polish", "one two three prompt four")


def test_without_any_rephrasing_the_text_is_delivered_unchanged():
    assert plan_rephrase("prompt write a poem", config(), "   ", "selection") is None
