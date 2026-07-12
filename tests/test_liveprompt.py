"""Tests for LivePrompt trigger detection/stripping in app/core/liveprompt.py."""
from __future__ import annotations

from app.core import liveprompt


class TestParseTriggerWords:
    def test_splits_and_lowercases(self):
        assert liveprompt.parse_trigger_words("Prompt, KI, ") == ["prompt", "ki"]

    def test_empty_input(self):
        assert liveprompt.parse_trigger_words("") == []
        assert liveprompt.parse_trigger_words(", ,") == []


class TestContainsTrigger:
    def test_trigger_within_scan_depth(self):
        assert liveprompt.contains_trigger("Prompt, schreibe ein Gedicht", ["prompt"], 5)

    def test_trigger_beyond_scan_depth_is_ignored(self):
        text = "eins zwei drei vier fünf prompt sechs"
        assert not liveprompt.contains_trigger(text, ["prompt"], 5)
        assert liveprompt.contains_trigger(text, ["prompt"], 6)

    def test_case_insensitive(self):
        assert liveprompt.contains_trigger("PROMPT do things", ["prompt"], 5)

    def test_no_triggers_configured(self):
        assert not liveprompt.contains_trigger("prompt do things", [], 5)


class TestStripTrigger:
    def test_strips_trigger_and_leading_punctuation(self):
        result = liveprompt.strip_trigger("Das bitte als Prompt. Schreib einen Witz", ["prompt"])
        assert result == "Schreib einen Witz"

    def test_earliest_trigger_end_wins(self):
        result = liveprompt.strip_trigger("ki: prompt schreib was", ["prompt", "ki"])
        assert result == "prompt schreib was"

    def test_no_trigger_returns_original(self):
        assert liveprompt.strip_trigger("nichts zu tun", ["prompt"]) == "nichts zu tun"

    def test_empty_remainder_falls_back_to_full_text(self):
        assert liveprompt.strip_trigger("prompt", ["prompt"]) == "prompt"
