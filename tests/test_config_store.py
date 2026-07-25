"""Tests for config loading, saving and migrations in app/core/config_store.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.config_store import ConfigStore
from app.core.constants import (
    CONFIG_SCHEMA_VERSION,
    DEFAULT_CONFIG,
    DEFAULT_REPHRASING_MODEL,
    PREVIOUS_DEFAULT_REPHRASING_MODELS,
    WINDOW_MIN_HEIGHT,
)
from app.core.hotkeys import normalize_hotkey_string


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
            assert config[key] == value

    def test_corrupt_file_yields_defaults(self, store: ConfigStore):
        with open(store.config_file, "w", encoding="utf-8") as f:
            f.write("{not valid json")
        config, changed = store.load()
        assert changed is True
        assert config["transcription_provider"] == DEFAULT_CONFIG["transcription_provider"]

    def test_existing_values_are_preserved(self, store: ConfigStore):
        write_config(store, {
            "provider_api_keys": {"openai": "sk-test", "groq": ""},
            "config_schema_version": CONFIG_SCHEMA_VERSION,
        })
        config, _ = store.load()
        assert config["provider_api_keys"]["openai"] == "sk-test"


class TestMigrations:
    def test_legacy_language_key_is_renamed(self, store: ConfigStore):
        write_config(store, {"language": "de"})
        config, changed = store.load()
        assert changed is True
        assert config["input_language"] == "de"
        assert "language" not in config

    def test_language_display_name_becomes_code(self, store: ConfigStore):
        write_config(store, {"input_language": "German"})
        config, _ = store.load()
        assert config["input_language"] == "de"

    def test_old_default_rephrasing_models_are_migrated(self, store: ConfigStore):
        for old_model in PREVIOUS_DEFAULT_REPHRASING_MODELS:
            write_config(store, {"rephrasing_model": old_model})
            config, _ = store.load()
            assert config["rephrasing_model"] == DEFAULT_REPHRASING_MODEL

    def test_user_chosen_rephrasing_model_is_kept(self, store: ConfigStore):
        write_config(store, {"rephrasing_model": "my-own-model"})
        config, _ = store.load()
        assert config["rephrasing_model"] == "my-own-model"

    def test_mojibake_in_text_fields_is_repaired(self, store: ConfigStore):
        write_config(store, {"prompt": "Ãœbersetze fÃ¼r mich"})
        config, _ = store.load()
        assert config["prompt"] == "Übersetze für mich"

    def test_mojibake_in_rephrase_entries_is_repaired(self, store: ConfigStore):
        write_config(store, {"post_rephrasing_entries": [{"caption": "HÃ¶flich", "text": "Sei hÃ¶flich"}]})
        config, _ = store.load()
        assert config["post_rephrasing_entries"][0]["caption"] == "Höflich"
        assert config["post_rephrasing_entries"][0]["text"] == "Sei höflich"

    def test_hotkey_with_control_chars_is_reset_to_default(self, store: ConfigStore):
        write_config(store, {"hotkey": "<ctrl>+\x03+<f9>+c"})
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

    def test_known_groq_credentials_are_deduplicated(self, store: ConfigStore):
        write_config(store, {
            "api_key": "gsk-one-key",
            "api_endpoint": "https://api.groq.com/openai/v1/audio/transcriptions",
            "rephrasing_api_url": "https://api.groq.com/openai/v1/chat/completions",
            "rephrasing_api_key": "gsk-one-key",
            "model": "whisper-large-v3-turbo (groq)",
            "rephrasing_model": "llama-3.3-70b-versatile",
        })
        config, _ = store.load()
        assert config["provider_api_keys"]["groq"] == "gsk-one-key"
        assert config["transcription_provider"] == "groq"
        assert config["rephrasing_provider"] == "groq"
        assert config["model"] == "whisper-large-v3-turbo"
        assert not {"api_key", "api_endpoint", "rephrasing_api_key", "rephrasing_api_url"} & config.keys()

    def test_untouched_secondary_defaults_reuse_transcription_provider(self, store: ConfigStore):
        write_config(store, {
            "api_key": "gsk-one-key",
            "api_endpoint": "https://api.groq.com/openai/v1/audio/transcriptions",
            "rephrasing_api_url": "https://api.openai.com/v1/chat/completions",
            "rephrasing_api_key": "",
            "rephrasing_model": DEFAULT_REPHRASING_MODEL,
        })
        config, _ = store.load()
        assert config["rephrasing_provider"] == "groq"
        assert config["rephrasing_model"] == "openai/gpt-oss-120b"

    def test_different_custom_endpoints_remain_separate(self, store: ConfigStore):
        write_config(store, {
            "api_key": "stt-key",
            "api_endpoint": "https://stt.example/v1/transcribe",
            "rephrasing_api_url": "https://chat.example/v1/chat",
            "rephrasing_api_key": "chat-key",
        })
        config, _ = store.load()
        assert config["transcription_provider"] == "custom"
        assert config["rephrasing_provider"] == "custom"
        assert config["custom_provider_settings"]["transcription"]["api_key"] == "stt-key"
        assert config["custom_provider_settings"]["rephrasing"]["api_key"] == "chat-key"

    def test_nested_defaults_are_not_shared_between_loads(self, store: ConfigStore, tmp_path: Path):
        first, _ = store.load()
        first["provider_api_keys"]["groq"] = "mutated"
        second_store = ConfigStore(str(tmp_path / "other.json"), normalize_hotkey_string)
        second, _ = second_store.load()
        assert second["provider_api_keys"]["groq"] == ""


class TestSaveRoundtrip:
    def test_utf8_roundtrip(self, store: ConfigStore):
        config, _ = store.load()
        config["prompt"] = "Bitte übersetze — dies ist ein Test 🚀"
        store.save(config)
        reloaded, changed = store.load()
        assert reloaded["prompt"] == "Bitte übersetze — dies ist ein Test 🚀"
        assert changed is False  # a fully migrated config must load unchanged
