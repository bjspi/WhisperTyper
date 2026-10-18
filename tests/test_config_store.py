"""Tests for config loading, saving and migrations in app/core/config_store.py."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.core.api_keys import selected_api_key
from app.core.config_store import ConfigStore
from app.core.constants import (
    CONFIG_SCHEMA_VERSION,
    DEFAULT_CONFIG,
    WINDOW_MIN_HEIGHT,
)
from app.core.env import is_MACOS
from app.core.hotkeys import normalize_hotkey_string
from app.core.prompts import default_instruction_entry


@pytest.fixture
def store(tmp_path: Path) -> ConfigStore:
    return ConfigStore(str(tmp_path / "config.json"), normalize_hotkey_string)


def write_config(store: ConfigStore, data: dict) -> None:
    with open(store.config_file, "w", encoding="utf-8") as f:
        json.dump(data, f)


class TestLoad:
    def test_missing_file_yields_full_defaults(self, store: ConfigStore):
        config, changed = store.load()
        assert changed is True
        for key, value in DEFAULT_CONFIG.items():
            if key != "post_rephrasing_entries":
                assert config[key] == value
        # The prompt list starts with the instruction entry in the UI language.
        assert config["post_rephrasing_entries"] == [default_instruction_entry(DEFAULT_CONFIG["ui_language"])]

    def test_an_unedited_earlier_default_prompt_becomes_the_current_default(self, store: ConfigStore):
        from app.core.prompts import DEFAULT_TRANSCRIPTION_PROMPTS, PREVIOUS_TRANSCRIPTION_PROMPTS
        write_config(store, {"prompt": PREVIOUS_TRANSCRIPTION_PROMPTS["de"]})
        assert store.load()[0]["prompt"] == DEFAULT_TRANSCRIPTION_PROMPTS["de"].strip()
        write_config(store, {"prompt": "Mein eigener Prompt."})
        assert store.load()[0]["prompt"] == "Mein eigener Prompt."

    @pytest.mark.parametrize("content", [
        b"{not valid json",
        '{"prompt": "führt"}'.encode("cp1252"),  # not UTF-8
        b'["a", "list"]',
    ])
    def test_unusable_file_yields_defaults_and_is_kept_aside(self, store: ConfigStore, content: bytes):
        with open(store.config_file, "wb") as f:
            f.write(content)
        config, changed = store.load()
        assert changed is True
        assert config["api_endpoint"] == DEFAULT_CONFIG["api_endpoint"]
        kept = [name for name in os.listdir(os.path.dirname(store.config_file)) if ".corrupt-" in name]
        assert len(kept) == 1 and kept[0].startswith("config.json.corrupt-")
        assert not os.path.exists(store.config_file)
        with open(os.path.join(os.path.dirname(store.config_file), kept[0]), "rb") as f:
            assert f.read() == content

    @pytest.mark.parametrize("key,value", [
        ("hotkey", None), ("post_rephrase_hotkey", 9), ("window_height", "abc"), ("window_width", None),
        ("prompt", ["x"]), ("post_rephrasing_entries", 3), ("config_schema_version", "3"),
    ])
    def test_wrong_typed_value_counts_as_missing(self, store: ConfigStore, key: str, value):
        write_config(store, {key: value, "api_key": "sk-keep"})
        config, changed = store.load()
        assert changed is True
        if key == "post_rephrasing_entries":
            assert config[key] == [default_instruction_entry(DEFAULT_CONFIG["ui_language"])]
        else:
            assert config[key] == DEFAULT_CONFIG[key]
        assert selected_api_key(config, "transcription") == "sk-keep"

    def test_existing_values_are_preserved(self, store: ConfigStore):
        write_config(store, {"api_key": "sk-test", "config_schema_version": CONFIG_SCHEMA_VERSION})
        config, _ = store.load()
        assert selected_api_key(config, "transcription") == "sk-test"
        assert "api_key" not in config


class TestMigrations:
    def test_legacy_language_key_is_renamed(self, store: ConfigStore):
        write_config(store, {"language": "de"})
        config, changed = store.load()
        assert changed is True
        assert config["input_language"] == "de"
        assert "language" not in config

    def test_clipboard_shortcut_hotkey_is_reset_to_default(self, store: ConfigStore):
        reserved = "<cmd>+c" if is_MACOS else "<ctrl>+c"
        write_config(store, {"post_rephrase_hotkey": reserved})
        config, changed = store.load()
        assert changed is True
        assert config["post_rephrase_hotkey"] == DEFAULT_CONFIG["post_rephrase_hotkey"]

    def test_global_generic_rephrasing_keys_are_dropped(self, store: ConfigStore):
        write_config(store, {"generic_rephrase_enabled": True, "generic_rephrase_prompt": "Polish"})
        config, changed = store.load()
        assert changed is True
        assert "generic_rephrase_enabled" not in config and "generic_rephrase_prompt" not in config

    def test_liveprompt_settings_become_the_first_prompt_entry(self, store: ConfigStore):
        write_config(store, {
            "ui_language": "en",
            "liveprompt_enabled": False, "liveprompt_trigger_words": "befehl, ki",
            "liveprompt_trigger_word_scan_depth": 7, "liveprompt_strip_trigger": True,
            "liveprompt_system_prompt": "My own instruction prompt", "rephrase_use_selection_context": True,
            "post_rephrasing_entries": [{"caption": "Polish", "text": "Polish it"}],
        })
        config, changed = store.load()
        assert changed is True
        instruction, template = config["post_rephrasing_entries"]
        assert instruction == {
            "kind": "instruction", "caption": "\u2728 Instruction", "text": "My own instruction prompt",
            # A switched-off LivePrompt keeps the entry usable by click; only the trigger word is off.
            "show_during_recording": False, "auto_apply": False, "enabled": True, "trigger_enabled": False,
            "trigger_words": "befehl, ki", "scan_depth": 7, "strip_trigger": True, "use_selection_context": True,
        }
        assert template["caption"] == "Polish"
        assert not any(key.startswith("liveprompt_") or key == "rephrase_use_selection_context" for key in config)

    def test_instruction_migration_runs_once(self, store: ConfigStore):
        config, _ = store.load()
        config["post_rephrasing_entries"].append({"caption": "Mail", "text": "Write a mail",
                                                  "show_during_recording": False})
        store.save(config)
        reloaded, changed = store.load()
        assert changed is False
        assert [entry.get("kind") for entry in reloaded["post_rephrasing_entries"]] == ["instruction", None]

    def test_windows_fast_paste_key_is_renamed(self, store: ConfigStore):
        write_config(store, {"windows_fast_paste": True})
        config, changed = store.load()
        assert changed is True
        assert config["fast_paste"] is True
        assert "windows_fast_paste" not in config

    def test_language_display_name_becomes_code(self, store: ConfigStore):
        write_config(store, {"input_language": "German"})
        config, _ = store.load()
        assert config["input_language"] == "de"


    def test_mojibake_in_text_fields_is_repaired(self, store: ConfigStore):
        write_config(store, {"prompt": "Ãœbersetze fÃ¼r mich"})
        config, _ = store.load()
        assert config["prompt"] == "Übersetze für mich"

    def test_mojibake_in_rephrase_entries_is_repaired(self, store: ConfigStore):
        write_config(store, {"post_rephrasing_entries": [{"caption": "HÃ¶flich", "text": "Sei hÃ¶flich"}]})
        config, _ = store.load()
        assert config["post_rephrasing_entries"][1]["caption"] == "Höflich"
        assert config["post_rephrasing_entries"][1]["text"] == "Sei höflich"

    def test_old_rephrase_entries_opt_out_of_recording_palette(self, store: ConfigStore):
        write_config(store, {"post_rephrasing_entries": [{"caption": "Polish", "text": "Improve it"}]})
        config, changed = store.load()
        assert changed is True
        assert config["post_rephrasing_entries"][0]["show_during_recording"] is False

    def test_invalid_recording_palette_flag_is_safely_disabled(self, store: ConfigStore):
        write_config(store, {
            "post_rephrasing_entries": [
                {"caption": "Polish", "text": "Improve it", "show_during_recording": "yes"},
            ],
        })
        config, changed = store.load()
        assert changed is True
        assert config["post_rephrasing_entries"][0]["show_during_recording"] is False

    def test_control_char_tokens_are_dropped_from_a_hotkey(self, store: ConfigStore):
        write_config(store, {"hotkey": "<ctrl>+\x03+<f9>+c"})
        config, _ = store.load()
        assert config["hotkey"] == "<ctrl>+<f9>+c"

    def test_hotkey_of_only_control_chars_falls_back_to_default(self, store: ConfigStore):
        write_config(store, {"hotkey": "\x03+\x16"})
        config, _ = store.load()
        assert config["hotkey"] == DEFAULT_CONFIG["hotkey"]

    def test_hotkeys_are_normalized_on_load(self, store: ConfigStore):
        write_config(store, {"hotkey": "Ctrl+F9"})
        config, _ = store.load()
        assert config["hotkey"] == "<ctrl>+<f9>"

    def test_schema_bump_raises_window_height(self, store: ConfigStore):
        write_config(store, {"window_height": 500})  # pre-schema config
        config, _ = store.load()
        assert config["window_height"] >= WINDOW_MIN_HEIGHT
        assert config["config_schema_version"] == CONFIG_SCHEMA_VERSION

    @pytest.mark.parametrize("schema", [0, CONFIG_SCHEMA_VERSION])
    def test_short_window_height_survives_reload_and_migration(self, store: ConfigStore, schema: int):
        config, _ = store.load()
        config["window_height"] = 640
        config["config_schema_version"] = schema
        store.save(config)
        reloaded, _ = store.load()
        assert reloaded["window_height"] == 640


class TestSaveRoundtrip:
    def test_interrupted_write_keeps_the_previous_file(self, store: ConfigStore, monkeypatch):
        config, _ = store.load()
        config["prompt"] = "kept"
        store.save(config)

        def crash(*_args, **_kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr("app.core.config_store.json.dump", crash)
        config["prompt"] = "lost"
        store.save(config)
        with open(store.config_file, encoding="utf-8") as f:
            assert json.load(f)["prompt"] == "kept"
        assert not os.path.exists(store.config_file + ".tmp")

    def test_save_creates_a_missing_directory(self, tmp_path: Path):
        store = ConfigStore(str(tmp_path / "new" / "dir" / "config.json"), normalize_hotkey_string)
        config, _ = store.load()
        store.save(config)
        assert store.load()[1] is False

    def test_utf8_roundtrip(self, store: ConfigStore):
        config, _ = store.load()
        config["prompt"] = "Bitte übersetze — dies ist ein Test 🚀"
        store.save(config)
        reloaded, changed = store.load()
        assert reloaded["prompt"] == "Bitte übersetze — dies ist ein Test 🚀"
        assert changed is False  # a fully migrated config must load unchanged

