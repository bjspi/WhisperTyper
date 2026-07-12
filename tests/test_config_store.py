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
        assert config["api_endpoint"] == DEFAULT_CONFIG["api_endpoint"]

    def test_existing_values_are_preserved(self, store: ConfigStore):
        write_config(store, {"api_key": "sk-test", "config_schema_version": CONFIG_SCHEMA_VERSION})
        config, _ = store.load()
        assert config["api_key"] == "sk-test"


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


class TestSaveRoundtrip:
    def test_utf8_roundtrip(self, store: ConfigStore):
        config, _ = store.load()
        config["prompt"] = "Bitte übersetze — dies ist ein Test 🚀"
        store.save(config)
        reloaded, changed = store.load()
        assert reloaded["prompt"] == "Bitte übersetze — dies ist ein Test 🚀"
        assert changed is False  # a fully migrated config must load unchanged
