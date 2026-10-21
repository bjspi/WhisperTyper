"""Saving reports failures, discarded settings edits never reach the disk, custom keys stay explicit."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.core import config_store as config_store_module
from app.core.api_keys import NO_KEY_PROFILE_ID, resolve_key_profile, selected_api_key
from app.core.config_store import ConfigStore

CUSTOM_PROFILES = [
    {"id": "a", "name": "Server A", "provider": "custom", "key": "key-for-a"},
    {"id": "b", "name": "Server B", "provider": "custom", "key": "key-for-b"},
]


@pytest.fixture
def store(tmp_path: Path) -> ConfigStore:
    return ConfigStore(str(tmp_path / "config.json"), normalize_hotkey=lambda text: text)


def test_a_briefly_locked_file_is_retried_and_a_lasting_lock_is_reported(store, monkeypatch):
    monkeypatch.setattr(config_store_module, "_REPLACE_RETRY_S", 0)
    real_replace = os.replace
    failures = [2]

    def replace(source, target):
        if failures[0] > 0:
            failures[0] -= 1
            raise PermissionError("held by the virus scanner")
        real_replace(source, target)

    monkeypatch.setattr(config_store_module.os, "replace", replace)
    assert store.save({"x": 1}) is True and store.last_error == ""
    assert json.loads(Path(store.config_file).read_text(encoding="utf-8")) == {"x": 1}

    failures[0] = 1000
    assert store.save({"x": 2}) is False and "virus scanner" in store.last_error
    assert json.loads(Path(store.config_file).read_text(encoding="utf-8")) == {"x": 1}
    assert not os.path.exists(store.config_file + ".tmp")


def test_unsaved_settings_edits_are_written_with_their_saved_values(store):
    pytest.importorskip("PyQt6.QtWidgets")
    from PyQt6.QtWidgets import QApplication

    from app.audio.store import RecordingStore
    from app.context import AppContext

    _app = QApplication.instance() or QApplication([])
    ctx = AppContext(store, RecordingStore(os.path.dirname(store.config_file)))
    ctx.config["ui_language"] = "fr"  # an edit in the open settings form, not saved yet
    ctx.set_unsaved_values(lambda: {"ui_language": "en"})
    ctx.config["input_device_name"] = "USB mic"  # e.g. picked in the palette, saved at once
    assert ctx.save_config() is True
    stored = json.loads(Path(store.config_file).read_text(encoding="utf-8"))
    assert stored["ui_language"] == "en" and stored["input_device_name"] == "USB mic"
    # The form's own Save writes its edits.
    assert ctx.save_config(with_unsaved_edits=True) is True
    assert json.loads(Path(store.config_file).read_text(encoding="utf-8"))["ui_language"] == "fr"


def test_a_custom_endpoint_never_falls_back_to_another_servers_key():
    config = {"api_endpoint": "http://server-b.local/v1/audio/transcriptions", "api_key_profiles": CUSTOM_PROFILES,
              "transcription_key_profile_id": ""}
    assert resolve_key_profile(config, "transcription") is None
    assert selected_api_key({**config, "transcription_key_profile_id": "b"}, "transcription") == "key-for-b"
    assert resolve_key_profile({**config, "transcription_key_profile_id": NO_KEY_PROFILE_ID},
                               "transcription")["id"] == NO_KEY_PROFILE_ID
    # Official providers keep their automatic first key.
    groq = {"api_endpoint": "https://api.groq.com/openai/v1/audio/transcriptions", "transcription_key_profile_id": "",
            "api_key_profiles": [{"id": "g", "name": "Groq", "provider": "groq", "key": "gsk-1"}]}
    assert selected_api_key(groq, "transcription") == "gsk-1"


def test_an_upgrade_pins_the_key_a_custom_endpoint_used_automatically(store):
    Path(store.config_file).write_text(json.dumps({
        "config_schema_version": 3, "api_key_profiles": CUSTOM_PROFILES,
        "api_endpoint": "http://server-a.local/v1/audio/transcriptions", "transcription_key_profile_id": "",
        "rephrasing_api_url": "https://api.openai.com/v1/chat/completions", "rephrasing_key_profile_id": "",
    }), encoding="utf-8")
    config, changed = store.load()
    assert changed and config["transcription_key_profile_id"] == "a"  # the key it used before, now explicit
    assert config["rephrasing_key_profile_id"] == ""  # official provider: stays automatic
    store.save({**config, "transcription_key_profile_id": ""})
    config, _changed = store.load()
    assert config["transcription_key_profile_id"] == ""  # pinned once: a later empty choice stays empty
