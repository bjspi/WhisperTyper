"""Provider-first API sections of the settings window (offscreen Qt, temporary config)."""
from __future__ import annotations

import json
from unittest.mock import Mock, patch

import pytest

QtWidgets = pytest.importorskip("PyQt6.QtWidgets")

from app.audio.store import RecordingStore  # noqa: E402
from app.context import AppContext  # noqa: E402
from app.core.config_store import ConfigStore  # noqa: E402
from app.core.hotkeys import normalize_hotkey_string  # noqa: E402
from app.ui.settings.window import SettingsWindow  # noqa: E402

CUSTOM_URL = "http://127.0.0.1:9/v1/audio/transcriptions"
PROFILES = [
    {"id": "g1", "name": "Groq one", "provider": "groq", "key": "gsk-test-one-0000"},
    {"id": "g2", "name": "Groq two", "provider": "groq", "key": "gsk-test-two-0000"},
    {"id": "c1", "name": "Server", "provider": "custom", "key": "custom-test-key"},
]


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def make_window(qapp, tmp_path):
    windows = []

    def build(**config):
        path = tmp_path / "config.json"
        if not path.exists():
            base = {"api_key_profiles": PROFILES, "ui_language": "en",
                    "api_endpoint": "https://api.groq.com/openai/v1/audio/transcriptions",
                    "rephrasing_api_url": "https://api.groq.com/openai/v1/chat/completions"}
            path.write_text(json.dumps({**base, **config}), encoding="utf-8")
        ctx = AppContext(ConfigStore(str(path), normalize_hotkey_string), RecordingStore(str(tmp_path)))
        recording = Mock(selectable_input_devices=Mock(return_value=[]))
        window = SettingsWindow(ctx, recording=recording, hotkeys=Mock(), files=Mock(log_file_exists=Mock(return_value=False)),
                                quit_app=Mock())
        windows.append(window)
        return window, path

    yield build
    for window in windows:
        window.deleteLater()


def save(window):
    with patch.object(window, "_collect_validation_warnings", return_value=[]):
        window.save_settings()


def test_official_provider_uses_the_first_key_and_hides_url_and_selector(make_window):
    window, _path = make_window()
    for task in ("transcription", "rephrasing"):
        widgets = window._task_widgets(task)
        assert widgets.endpoint.isHidden() and widgets.key_profile.isHidden()
        assert widgets.key_status.property("key_status") == "ok" and "Groq one" in widgets.key_status.text()
        assert not widgets.key_choose.isHidden() and widgets.key_add.isHidden()
    assert window._ui_api_key("transcription") == "gsk-test-one-0000"


def test_missing_key_warns_and_offers_the_api_keys_tab(make_window):
    window, _path = make_window()
    window.transcription_provider_selector.setCurrentIndex(window.transcription_provider_selector.findData("openai"))
    assert window.transcription_key_status_label.property("key_status") == "missing"
    assert window.transcription_api_group.property("incomplete") is True
    window.transcription_key_add_button.click()
    assert window.tabs.currentWidget() is window._api_keys_tab


def test_custom_url_survives_switching_to_an_official_provider(make_window):
    window, path = make_window()
    window.transcription_provider_selector.setCurrentIndex(window.transcription_provider_selector.findData("custom"))
    assert not window.api_endpoint_input.isHidden() and not window.transcription_key_profile_selector.isHidden()
    window.api_endpoint_input.setText(CUSTOM_URL)
    assert window._ui_api_key("transcription") == "custom-test-key"
    window.transcription_provider_selector.setCurrentIndex(window.transcription_provider_selector.findData("groq"))
    window.transcription_provider_selector.setCurrentIndex(window.transcription_provider_selector.findData("custom"))
    assert window.api_endpoint_input.text() == CUSTOM_URL
    window.transcription_provider_selector.setCurrentIndex(window.transcription_provider_selector.findData("groq"))
    save(window)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["transcription_custom_url"] == CUSTOM_URL
    assert saved["api_endpoint"] == "https://api.groq.com/openai/v1/audio/transcriptions"


def test_chosen_key_persists_until_the_provider_changes(make_window):
    window, path = make_window()
    window.rephrasing_key_choose_button.click()
    assert not window.rephrasing_key_profile_selector.isHidden()
    window.rephrasing_key_profile_selector.setCurrentIndex(window.rephrasing_key_profile_selector.findData("g2"))
    assert window._ui_api_key("rephrasing") == "gsk-test-two-0000"
    save(window)
    assert json.loads(path.read_text(encoding="utf-8"))["rephrasing_key_profile_id"] == "g2"

    reloaded, _path = make_window()
    assert not reloaded.rephrasing_key_profile_selector.isHidden()  # differs from the automatic key
    assert reloaded._ui_api_key("rephrasing") == "gsk-test-two-0000"
    reloaded.rephrasing_provider_selector.setCurrentIndex(reloaded.rephrasing_provider_selector.findData("openai"))
    reloaded.rephrasing_provider_selector.setCurrentIndex(reloaded.rephrasing_provider_selector.findData("groq"))
    assert reloaded.rephrasing_key_profile_selector.currentData() == ""
    assert reloaded.rephrasing_key_profile_selector.isHidden()
    assert reloaded._ui_api_key("rephrasing") == "gsk-test-one-0000"


def test_both_api_sections_have_the_same_controls(make_window):
    window, _path = make_window()
    transcription, rephrasing = (window._task_widgets(task) for task in ("transcription", "rephrasing"))
    assert [type(widget) for widget in transcription] == [type(widget) for widget in rephrasing]
    assert window.test_transcription_api_button.text() == window.test_rephrasing_api_button.text()
