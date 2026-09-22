"""Tests for TranslationManager (real language files) and default-prompt helpers."""
from __future__ import annotations

import json
import os

from app.core.constants import UI_LANG_FILES
from app.core.i18n import TranslationManager
from app.core.paths import resource_path
from app.core.prompts import (
    DEFAULT_TRANSCRIPTION_PROMPTS,
    _default_prompt_for,
    _is_known_default_prompt,
    recording_prompt_entries,
)


class TestTranslationManager:
    def test_loads_english(self):
        tr = TranslationManager("en")
        assert tr.language == "en"
        assert tr.translations  # non-empty

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
            {"caption": "First", "text": "Do first"},
            {"caption": "Second", "text": "Do second"},
        ]

    def test_rejects_non_lists_and_non_boolean_opt_in(self):
        assert recording_prompt_entries(None) == []
        assert recording_prompt_entries([
            {"caption": "Wrong type", "text": "No", "show_during_recording": 1},
        ]) == []
