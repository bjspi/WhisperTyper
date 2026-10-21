"""Closing the settings window without Save: live-mirrored edits are dropped, only the size is persisted."""
from __future__ import annotations

import json
import threading
import time
from unittest.mock import Mock, patch

import pytest

QtWidgets = pytest.importorskip("PyQt6.QtWidgets")

from app.audio.store import RecordingStore  # noqa: E402
from app.context import AppContext  # noqa: E402
from app.core.config_store import ConfigStore  # noqa: E402
from app.core.hotkeys import normalize_hotkey_string  # noqa: E402
from app.ui.settings.window import SettingsWindow  # noqa: E402

PROFILES = [{"id": "g1", "name": "Groq one", "provider": "groq", "key": "gsk-test-one-0000"}]


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def make_window(qapp, tmp_path):
    windows = []

    def build(**config):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"ui_language": "en", "api_key_profiles": PROFILES, **config}), encoding="utf-8")
        ctx = AppContext(ConfigStore(str(path), normalize_hotkey_string), RecordingStore(str(tmp_path)))
        window = SettingsWindow(ctx, recording=Mock(selectable_input_devices=Mock(return_value=[])), hotkeys=Mock(),
                                files=Mock(log_file_exists=Mock(return_value=False)), quit_app=Mock())
        windows.append(window)
        return window, path

    yield build
    for window in windows:
        window.deleteLater()


def on_disk(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_closing_without_save_drops_prompt_edits_and_leaves_the_file_alone(make_window):
    window, path = make_window()
    before = on_disk(path)
    saved_entries = json.loads(json.dumps(window.config["post_rephrasing_entries"]))
    count = window.post_rp_list.count()
    window.post_rp_add_btn.click()
    window.post_rp_caption_edit.setText("Unsaved caption")
    window._sync_transformations_to_config()
    assert window.config["post_rephrasing_entries"] != saved_entries  # mirrored live for the palettes
    window.close()
    after = on_disk(path)
    # The stored width leaves out the modern sidebar, so both designs open with equally wide pages.
    assert after["window_width"] == window.width() - window.design_extra_width()
    assert after["window_height"] == window.height()
    assert {k: v for k, v in after.items() if k not in ("window_width", "window_height")} == \
        {k: v for k, v in before.items() if k not in ("window_width", "window_height")}
    assert window.config["post_rephrasing_entries"] == saved_entries
    assert window.post_rp_list.count() == count
    assert all("Unsaved caption" != entry.get("caption") for entry in window._transformations.entries())


def test_closing_without_save_restores_the_ui_language(make_window):
    window, path = make_window()
    window.ui_language_selector.setCurrentIndex(window.ui_language_selector.findData("de"))
    assert window.config["ui_language"] == "de" and window.translator.language == "de"
    window.close()
    assert window.config["ui_language"] == "en" and window.translator.language == "en"
    assert window.ui_language_selector.currentData() == "en"
    assert window.windowTitle() == window.translator.tr("window_title")
    assert on_disk(path)["ui_language"] == "en"


def test_save_makes_the_edits_the_new_baseline_for_a_later_close(make_window):
    window, path = make_window()
    window.post_rp_add_btn.click()
    window.post_rp_caption_edit.setText("Kept caption")
    with patch.object(window, "_collect_validation_warnings", return_value=[]):
        window.save_settings()
    window.close()
    captions = [entry["caption"] for entry in on_disk(path)["post_rephrasing_entries"]]
    assert "Kept caption" in captions
    assert "Kept caption" in [entry["caption"] for entry in window.config["post_rephrasing_entries"]]


def test_ffmpeg_probe_runs_once_in_the_background_and_retranslates_from_its_cache(make_window, qapp):
    probing_threads = []

    def fake_probe(_exe):
        probing_threads.append(threading.current_thread())
        return "7.1-test"

    with patch("app.ui.settings.transcription_page.probe_version", side_effect=fake_probe) as probe:
        window, _path = make_window()
        assert window.ffmpeg_status_label.text() == ""  # nothing until the probe reports
        deadline = time.monotonic() + 5
        while not window._ffmpeg_probed and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        assert window.ffmpeg_status_label.text() == window.translator.tr("ffmpeg_status_detected", version="7.1-test")
        assert probe.call_count == 1 and probing_threads[0] is not threading.main_thread()
        window.change_language("de")
        assert "FFmpeg erkannt: 7.1-test" in window.ffmpeg_status_label.text()
        assert probe.call_count == 1  # retranslating only re-renders the cached result
        window.ffmpeg_path_input.setText("C:/nowhere/ffmpeg.exe")
        deadline = time.monotonic() + 5
        while probe.call_count < 2 and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        assert probe.call_count == 2  # a typed path is probed again after the debounce pause
