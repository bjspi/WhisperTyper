"""Tests for TranslationManager (real language files) and default-prompt helpers."""
from __future__ import annotations

import json
import os

from app.core.constants import UI_LANG_FILES
from app.core.i18n import TranslationManager
from app.core.paths import resource_path
from app.core.prompts import (
    DEFAULT_TRANSCRIPTION_PROMPTS,
    MAX_TRANSFORMATIONS,
    _default_prompt_for,
    _is_known_default_prompt,
    auto_apply_prompt,
    captioned_transformations,
    instruction_entry,
    load_transformations,
    recording_prompt_entries,
)


class TestTranslationManager:

    def test_unknown_language_falls_back_to_english(self):
        tr = TranslationManager("xx")
        assert tr.language == "en"

    def test_missing_key_returns_key(self):
        tr = TranslationManager("en")
        assert tr.tr("definitely_not_a_key") == "definitely_not_a_key"

    def test_placeholder_formatting(self):
        tr = TranslationManager("en")
        tr.translations["_test_key"] = "Hotkey is {hotkey}"
        assert tr.tr("_test_key", hotkey="F9") == "Hotkey is F9"

    def test_malformed_placeholder_never_raises(self):
        tr = TranslationManager("en")
        tr.translations["_bad_key"] = "Broken {placeholder"
        assert tr.tr("_bad_key", placeholder="x") == "Broken {placeholder"

    def test_all_language_files_are_valid_json_with_same_keys(self):
        lang_dir = resource_path("lang")
        reference_keys = None
        for lang in sorted(UI_LANG_FILES):
            with open(os.path.join(lang_dir, f"{lang}.json"), encoding="utf-8") as f:
                data = json.load(f)
            keys = set(data.keys())
            if reference_keys is None:
                reference_keys = keys
            else:
                missing = reference_keys - keys
                extra = keys - reference_keys
                assert not missing and not extra, (
                    f"{lang}.json key mismatch — missing: {sorted(missing)}, extra: {sorted(extra)}"
                )


class TestDefaultPrompts:
    def test_known_language(self):
        assert _default_prompt_for(DEFAULT_TRANSCRIPTION_PROMPTS, "de") == DEFAULT_TRANSCRIPTION_PROMPTS["de"].strip()

    def test_unknown_language_falls_back_to_english(self):
        assert _default_prompt_for(DEFAULT_TRANSCRIPTION_PROMPTS, "xx") == DEFAULT_TRANSCRIPTION_PROMPTS["en"].strip()

    def test_default_detection_across_languages(self):
        assert _is_known_default_prompt(DEFAULT_TRANSCRIPTION_PROMPTS, DEFAULT_TRANSCRIPTION_PROMPTS["fr"])
        assert not _is_known_default_prompt(DEFAULT_TRANSCRIPTION_PROMPTS, "my custom prompt")


class TestRecordingPromptEntries:
    def test_returns_only_enabled_complete_entries_in_order(self):
        entries = [
            {"caption": "First", "text": "Do first", "show_during_recording": True},
            {"caption": "Disabled", "text": "Do not show", "show_during_recording": False},
            {"caption": "", "text": "Missing caption", "show_during_recording": True},
            {"caption": "Missing text", "text": "  ", "show_during_recording": True},
            {"caption": "Second", "text": "Do second", "show_during_recording": True},
        ]
        assert recording_prompt_entries(entries) == [
            {"caption": "First", "text": "Do first", "auto_apply": False, "kind": "prompt"},
            {"caption": "Second", "text": "Do second", "auto_apply": False, "kind": "prompt"},
        ]

    def test_rejects_non_lists_and_non_boolean_opt_in(self):
        assert recording_prompt_entries(None) == []
        assert recording_prompt_entries([
            {"caption": "Wrong type", "text": "No", "show_during_recording": 1},
        ]) == []


class TestAutoApplyPrompt:
    def test_only_the_first_marked_prompt_applies_and_is_always_shown(self):
        entries = load_transformations([
            {"caption": "Fix", "text": "Fix it", "auto_apply": True, "show_during_recording": False},
            {"caption": "Mail", "text": "Write a mail", "auto_apply": True, "show_during_recording": True},
        ])
        assert [(entry["auto_apply"], entry["show_during_recording"]) for entry in entries
                if entry["kind"] == "prompt"] == [(True, True), (False, True)]
        assert auto_apply_prompt(entries) == "Fix it"
        assert [prompt["auto_apply"] for prompt in recording_prompt_entries(entries)] == [True, False]

    def test_no_marked_prompt_means_no_automatic_rephrasing(self):
        assert auto_apply_prompt([{"caption": "Fix", "text": "Fix it", "show_during_recording": True}]) is None
        assert auto_apply_prompt([{"caption": "", "text": "No caption", "auto_apply": True}]) is None
        assert auto_apply_prompt(None) is None


class TestInstructionEntry:
    INSTRUCTION = {"kind": "instruction", "caption": "Go", "text": "Carry it out", "show_during_recording": True}

    def test_exactly_one_instruction_entry_keeps_its_position(self):
        entries = load_transformations([{"caption": "A", "text": "a"}, self.INSTRUCTION, dict(self.INSTRUCTION)])
        assert [entry["kind"] for entry in entries] == ["prompt", "instruction"]
        assert load_transformations([{"caption": "A", "text": "a"}])[0]["kind"] == "instruction"  # added when missing

    def test_limit_counts_only_own_templates(self):
        templates = [{"caption": f"P{i}", "text": "t"} for i in range(MAX_TRANSFORMATIONS + 3)]
        entries = load_transformations([self.INSTRUCTION, *templates])
        assert sum(entry["kind"] == "prompt" for entry in entries) == MAX_TRANSFORMATIONS
        assert entries[0]["kind"] == "instruction"

    def test_instruction_is_offered_only_while_active(self):
        active = [self.INSTRUCTION, {"caption": "A", "text": "a", "show_during_recording": True}]
        inactive = [{**self.INSTRUCTION, "enabled": False}, active[1]]
        assert [entry["kind"] for entry in recording_prompt_entries(active)] == ["instruction", "prompt"]
        assert [entry["kind"] for entry in captioned_transformations(active)] == ["instruction", "prompt"]
        assert [entry["kind"] for entry in recording_prompt_entries(inactive)] == ["prompt"]
        assert [entry["kind"] for entry in captioned_transformations(inactive)] == ["prompt"]

    def test_instruction_never_applies_automatically(self):
        entry = instruction_entry([{**self.INSTRUCTION, "auto_apply": True}])
        assert entry["auto_apply"] is False
        assert auto_apply_prompt([{**self.INSTRUCTION, "auto_apply": True}]) is None

