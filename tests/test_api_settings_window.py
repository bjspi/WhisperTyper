"""Provider-first API sections of the settings window (offscreen Qt, temporary config)."""
from __future__ import annotations

import json
from unittest.mock import Mock, patch

import pytest

QtWidgets = pytest.importorskip("PyQt6.QtWidgets")

from app.audio.store import RecordingStore  # noqa: E402
from app.context import AppContext  # noqa: E402
from app.controllers.local_engine import LOCAL_ENGINE_SUPPORTED  # noqa: E402
from app.core.api_keys import LOCAL_ENDPOINT  # noqa: E402
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
    # A custom server gets no automatic key (it may belong to another server): it must be chosen.
    assert window._ui_api_key("transcription") == ""
    assert window.transcription_key_status_label.property("key_status") == "missing"
    selector = window.transcription_key_profile_selector
    selector.setCurrentIndex(selector.findData("c1"))
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
    assert window.rephrasing_key_profile_selector.isHidden()  # saving closes the selector

    reloaded, _path = make_window()
    assert reloaded.rephrasing_key_profile_selector.isHidden()
    assert reloaded._ui_api_key("rephrasing") == "gsk-test-two-0000"  # the choice itself is kept
    reloaded.rephrasing_provider_selector.setCurrentIndex(reloaded.rephrasing_provider_selector.findData("openai"))
    reloaded.rephrasing_provider_selector.setCurrentIndex(reloaded.rephrasing_provider_selector.findData("groq"))
    assert reloaded.rephrasing_key_profile_selector.currentData() == ""
    assert reloaded.rephrasing_key_profile_selector.isHidden()
    assert reloaded._ui_api_key("rephrasing") == "gsk-test-one-0000"


def test_choosing_the_automatic_key_explicitly_is_stored_as_automatic(make_window):
    window, path = make_window()
    window.rephrasing_key_choose_button.click()
    window.rephrasing_key_profile_selector.setCurrentIndex(window.rephrasing_key_profile_selector.findData("g1"))
    save(window)
    assert json.loads(path.read_text(encoding="utf-8"))["rephrasing_key_profile_id"] == ""
    assert window.rephrasing_key_profile_selector.isHidden()



def test_connection_test_runs_off_the_gui_thread_and_ignores_clicks_meanwhile(make_window, qapp):
    import threading

    window, _path = make_window()
    tester = window._connection_tester
    button = window.test_transcription_api_button
    caption = button.text()
    release, probe_thread = threading.Event(), []

    def slow_probe(*_args):
        probe_thread.append(threading.current_thread())
        release.wait(5)
        return "ok", ""

    with (patch("app.ui.connection_tester.net.run_transcription_connection_test", side_effect=slow_probe),
          patch("app.ui.connection_tester.request") as internet,
          patch("app.ui.connection_tester.message_dialog") as message_box):
        tester.test_transcription()
        assert tester.busy and not button.isEnabled()
        assert button.text() == window.translator.tr("api_test_testing_button")
        tester.test_internet()  # one test at a time: ignored while the first one runs
        assert window.test_internet_button.isEnabled() and not internet.called
        release.set()
        tester.wait()
    assert probe_thread and probe_thread[0] is not threading.main_thread()
    assert not tester.busy and button.isEnabled() and button.text() == caption
    assert message_box.inform.call_count == 1
    assert message_box.inform.call_args.args[1] == window.translator.tr("conn_test_title")


def test_a_tray_provider_switch_saves_only_the_switch_and_keeps_the_chosen_custom_key(make_window):
    window, path = make_window(transcription_custom_url=CUSTOM_URL, transcription_key_profile_id="c1")
    switched = Mock()
    window.transcription_provider_switched.connect(switched)
    window.prompt_input.setPlainText("an unsaved prompt edit")
    window.select_transcription_provider("custom")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["api_endpoint"] == CUSTOM_URL
    assert saved.get("prompt") != "an unsaved prompt edit"
    # The form follows, so a later Save keeps the switch and the custom server's key.
    assert window.api_endpoint_input.text() == CUSTOM_URL
    assert window._ui_api_key("transcription") == "custom-test-key"
    switched.assert_called_once()


@pytest.mark.skipif(not LOCAL_ENGINE_SUPPORTED, reason="no local engine on this platform")
def test_a_tray_switch_to_local_turns_the_engine_on_and_saves_that(make_window):
    window, path = make_window(local_engine_enabled=False)
    with patch.object(window._local_engine_tab, "_ensure_packages", return_value=True):
        assert window.select_transcription_provider("local")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["api_endpoint"] == LOCAL_ENDPOINT and saved["local_engine_enabled"] is True


def test_a_declined_engine_install_keeps_the_provider(make_window):
    window, path = make_window(local_engine_enabled=False)
    with patch.object(window._local_engine_tab, "_ensure_packages", return_value=False):
        assert not window.select_transcription_provider("local")
    assert json.loads(path.read_text(encoding="utf-8"))["api_endpoint"].startswith("https://api.groq.com/")
