"""Regression tests for rich clipboard capture and restoration."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from PyQt6.QtCore import QByteArray, QMimeData
from PyQt6.QtGui import QColor, QImage

import app.mixins.clipboard_mixin as clipboard_module
from app.core.timing import OperationTiming
from app.mixins.clipboard_mixin import ClipboardMixin


class FakeClipboard:
    """Small in-memory stand-in for QClipboard that retains QMimeData ownership."""

    def __init__(self, mime_data: QMimeData) -> None:
        """Initialize the fake with the supplied MIME payload."""
        self._mime_data = mime_data

    def mimeData(self) -> QMimeData:
        return self._mime_data

    def setMimeData(self, mime_data: QMimeData) -> None:
        self._mime_data = mime_data

    def clear(self) -> None:
        self._mime_data = QMimeData()


class ClipboardHarness(ClipboardMixin):
    def __init__(self) -> None:
        """Initialize the clipboard state and controllable restore timer."""
        self._pending_clipboard_restore_state = None
        self.config = {"restore_clipboard": True}
        self.on_simulated_key = lambda _char: None
        self._clipboard_restore_timer = FakeTimer(self._perform_clipboard_restore)

    def _check_and_warn_macos_permissions(self, _permission: str) -> None:
        """Avoid application-only permission UI in focused clipboard tests."""

    def _simulate_key_combination(self, char: str) -> bool:
        """Expose simulated copy/paste keys to each test without touching the OS."""
        self.on_simulated_key(char)
        return True


class FakeTimer:
    """Controllable single-shot timer for asynchronous restore tests."""

    def __init__(self, callback: Any) -> None:
        """Store the callback without starting the timer."""
        self.callback = callback
        self.delay_ms: int | None = None
        self.active = False

    def stop(self) -> None:
        self.active = False

    def start(self, delay_ms: int) -> None:
        self.delay_ms = delay_ms
        self.active = True

    def fire(self) -> None:
        assert self.active
        self.active = False
        self.callback()


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> tuple[ClipboardHarness, FakeClipboard]:
    fake_clipboard = FakeClipboard(QMimeData())

    class FakeApplication:
        @staticmethod
        def clipboard() -> FakeClipboard:
            return fake_clipboard

        @staticmethod
        def processEvents() -> None:
            pass

    monkeypatch.setattr(clipboard_module, "QApplication", FakeApplication)
    monkeypatch.setattr(clipboard_module, "is_MACOS", False)
    monkeypatch.setattr(clipboard_module.time, "sleep", lambda _seconds: None)

    def copy_text(text: str) -> None:
        mime_data = QMimeData()
        mime_data.setText(text)
        fake_clipboard.setMimeData(mime_data)

    monkeypatch.setattr(clipboard_module.copykitten, "copy", copy_text)
    monkeypatch.setattr(clipboard_module.copykitten, "clear", fake_clipboard.clear)
    monkeypatch.setattr(clipboard_module.copykitten, "paste", lambda: fake_clipboard.mimeData().text())
    return ClipboardHarness(), fake_clipboard


def _make_test_image() -> QImage:
    image = QImage(4, 3, QImage.Format.Format_ARGB32)
    image.fill(QColor(17, 91, 203, 177))
    return image


def test_full_clipboard_roundtrip_preserves_image_and_binary_formats(
    harness: tuple[ClipboardHarness, FakeClipboard],
) -> None:
    mixin, clipboard = harness
    original_mime = QMimeData()
    original_mime.setHtml("<p><b>rich text</b></p>")
    original_mime.setData("application/x-whispertyper-test", QByteArray(b"\x00\xffblob\x00"))
    original_mime.setImageData(_make_test_image())
    clipboard.setMimeData(original_mime)

    state = mixin._capture_clipboard_state()
    clipboard.setMimeData(QMimeData())
    clipboard.mimeData().setText("temporary transcription")
    mixin._restore_clipboard_state(state)

    restored = clipboard.mimeData()
    assert restored.html() == "<p><b>rich text</b></p>"
    assert bytes(restored.data("application/x-whispertyper-test")) == b"\x00\xffblob\x00"
    assert restored.hasImage()
    restored_image: Any = restored.imageData()
    assert isinstance(restored_image, QImage)
    assert restored_image.size() == _make_test_image().size()
    assert restored_image.pixelColor(0, 0) == QColor(17, 91, 203, 177)


def test_image_pixels_are_captured_when_raw_qt_image_payload_is_empty(
    harness: tuple[ClipboardHarness, FakeClipboard],
) -> None:
    mixin, clipboard = harness
    image_mime = QMimeData()
    image_mime.setImageData(_make_test_image())
    clipboard.setMimeData(image_mime)

    assert bytes(image_mime.data("application/x-qt-image")) == b""

    snapshot = mixin._capture_qt_clipboard_state()
    assert snapshot is not None
    assert isinstance(snapshot.get("image"), QImage)

    clipboard.clear()
    assert mixin._restore_qt_clipboard_state(snapshot) is True
    restored_image: Any = clipboard.mimeData().imageData()
    assert isinstance(restored_image, QImage)
    assert restored_image.pixelColor(2, 1) == QColor(17, 91, 203, 177)


def test_empty_clipboard_roundtrip_stays_empty(
    harness: tuple[ClipboardHarness, FakeClipboard],
) -> None:
    mixin, clipboard = harness

    state = mixin._capture_clipboard_state()
    temporary = QMimeData()
    temporary.setText("temporary transcription")
    clipboard.setMimeData(temporary)
    mixin._restore_clipboard_state(state)

    assert clipboard.mimeData().formats() == []


def test_insert_path_keeps_text_until_delayed_restore_even_when_paste_simulation_fails(
    harness: tuple[ClipboardHarness, FakeClipboard],
) -> None:
    mixin, clipboard = harness
    original_mime = QMimeData()
    original_mime.setImageData(_make_test_image())
    clipboard.setMimeData(original_mime)

    def fail_on_paste(char: str) -> None:
        assert char == "v"
        assert clipboard.mimeData().text() == "temporary transcription"
        raise RuntimeError("synthetic paste failure")

    mixin.on_simulated_key = fail_on_paste
    mixin.insert_transcribed_text("temporary transcription")

    assert clipboard.mimeData().text() == "temporary transcription"
    assert mixin._clipboard_restore_timer.delay_ms == 500
    assert mixin._pending_clipboard_restore_state is not None

    mixin._clipboard_restore_timer.fire()
    restored_image: Any = clipboard.mimeData().imageData()
    assert isinstance(restored_image, QImage)
    assert restored_image.pixelColor(1, 1) == QColor(17, 91, 203, 177)


def test_selected_text_path_restores_non_text_clipboard(
    harness: tuple[ClipboardHarness, FakeClipboard],
) -> None:
    mixin, clipboard = harness
    original_mime = QMimeData()
    original_mime.setData("application/x-whispertyper-test", QByteArray(b"original blob"))
    original_mime.setImageData(_make_test_image())
    clipboard.setMimeData(original_mime)

    def copy_selection(char: str) -> None:
        assert char == "c"
        selected_mime = QMimeData()
        selected_mime.setText("  selected for rephrasing  ")
        clipboard.setMimeData(selected_mime)

    mixin.on_simulated_key = copy_selection
    selected = mixin.get_selected_text()

    assert selected == "selected for rephrasing"
    assert bytes(clipboard.mimeData().data("application/x-whispertyper-test")) == b"original blob"
    assert clipboard.mimeData().hasImage()


def test_commit_timings_separate_paste_dispatch_from_existing_waits(harness, monkeypatch):
    mixin, clipboard = harness
    clock = [1_000_000_000]
    monkeypatch.setattr("app.core.timing.time.perf_counter_ns", lambda: clock[0])
    monkeypatch.setattr(clipboard_module.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + int(seconds * 1e9)))
    timing = OperationTiming("recording")
    assert mixin.insert_transcribed_text("private transcript", timing=timing)
    events = timing._events
    assert events["paste_dispatch_start"] - events["clipboard_write_end"] == 100_000_000
    assert events["paste_settle_end"] - events["paste_dispatch_end"] == 100_000_000
    assert events["text_commit_end"] == events["paste_dispatch_end"]
    assert events["text_commit_end"] < events["paste_settle_end"]
    assert "text_commit_failed" not in events
    assert clipboard.mimeData().text() == "private transcript"
    assert mixin._clipboard_restore_timer.active


@pytest.mark.parametrize("failure", ["copy", "paste_exception", "paste_false"])
def test_failed_commit_does_not_invent_success_and_preserves_restore(harness, monkeypatch, failure):
    mixin, clipboard = harness
    original = QMimeData()
    original.setText("original clipboard")
    clipboard.setMimeData(original)

    def fail(*_args):
        raise RuntimeError("synthetic failure")

    if failure == "copy":
        monkeypatch.setattr(clipboard_module.copykitten, "copy", fail)
    elif failure == "paste_exception":
        mixin.on_simulated_key = fail
    else:
        monkeypatch.setattr(mixin, "_simulate_key_combination", lambda _char: False)
    timing = OperationTiming("recording")
    assert not mixin.insert_transcribed_text("private transcript", timing=timing)
    assert "text_commit_failed" in timing._events
    assert "text_commit_end" not in timing._events
    assert "paste_dispatch_end" not in timing._events
    assert mixin._pending_clipboard_restore_state is not None


def test_empty_text_has_no_commit_milestones(harness):
    mixin, _clipboard = harness
    timing = OperationTiming("recording")
    assert mixin.insert_transcribed_text("", timing=timing)
    assert "text_commit_start" not in timing._events


def test_direct_input_preserves_rich_clipboard_and_has_no_paste_waits(harness, monkeypatch):
    mixin, clipboard = harness
    mixin.config["windows_sendinput_text"] = True
    monkeypatch.setattr(clipboard_module, "is_WINDOWS", True)
    send = Mock(return_value=(8, 8))
    monkeypatch.setattr(clipboard_module, "send_unicode_text", send)
    original = QMimeData()
    original.setImageData(_make_test_image())
    clipboard.setMimeData(original)
    copy = Mock(side_effect=AssertionError("Direct input must not touch clipboard"))
    monkeypatch.setattr(clipboard_module.copykitten, "copy", copy)
    monkeypatch.setattr(mixin, "_capture_clipboard_state", copy)
    monkeypatch.setattr(clipboard_module.time, "sleep", copy)
    monkeypatch.setattr(mixin, "_simulate_key_combination", copy)
    timing = OperationTiming("recording")
    assert mixin.insert_transcribed_text("ä🦄文", timing)
    send.assert_called_once_with("ä🦄文")
    assert clipboard.mimeData() is original
    assert not mixin._clipboard_restore_timer.active
    assert "sendinput_dispatch_end" in timing._events
    assert "text_commit_end" in timing._events
    assert "paste_dispatch_start" not in timing._events
    assert "clipboard_snapshot_start" not in timing._events


@pytest.mark.parametrize("rejection", [(0, 8), OSError("modifier held"), ValueError("control character")])
def test_direct_rejection_can_fallback_to_existing_paste(harness, monkeypatch, rejection):
    mixin, clipboard = harness
    mixin.config.update(windows_sendinput_text=True, windows_sendinput_fallback=True)
    monkeypatch.setattr(clipboard_module, "is_WINDOWS", True)
    send = Mock(side_effect=rejection) if isinstance(rejection, Exception) else Mock(return_value=rejection)
    monkeypatch.setattr(clipboard_module, "send_unicode_text", send)
    paste = Mock()
    mixin.on_simulated_key = paste
    timing = OperationTiming("recording")
    assert mixin.insert_transcribed_text("text", timing)
    paste.assert_called_once_with("v")
    assert clipboard.mimeData().text() == "text"
    assert mixin._clipboard_restore_timer.active
    assert "sendinput_fallback" in timing._events
    assert "paste_dispatch_end" in timing._events
    assert "text_commit_end" in timing._events
    assert "text_commit_failed" not in timing._events


@pytest.mark.parametrize("result,fallback", [((1, 8), True), ((3, 8), True), ((0, 8), False), (RuntimeError(), True)])
def test_partial_or_unknown_failure_never_duplicates_text(harness, monkeypatch, result, fallback):
    mixin, clipboard = harness
    original = clipboard.mimeData()
    mixin.config.update(windows_sendinput_text=True, windows_sendinput_fallback=fallback)
    monkeypatch.setattr(clipboard_module, "is_WINDOWS", True)
    send = Mock(side_effect=result) if isinstance(result, Exception) else Mock(return_value=result)
    monkeypatch.setattr(clipboard_module, "send_unicode_text", send)
    paste = Mock()
    mixin.on_simulated_key = paste
    mixin.translator = SimpleNamespace(tr=lambda key: key)
    mixin.show_tray_balloon = Mock()
    timing = OperationTiming("recording")
    assert not mixin.insert_transcribed_text("text", timing)
    paste.assert_not_called()
    mixin.show_tray_balloon.assert_called_once()
    assert clipboard.mimeData() is original
    assert not mixin._clipboard_restore_timer.active
    assert "text_commit_failed" in timing._events
    assert "text_commit_end" not in timing._events
    assert "sendinput_fallback" not in timing._events


def test_saved_windows_option_has_no_effect_off_windows(harness, monkeypatch):
    mixin, clipboard = harness
    mixin.config["windows_sendinput_text"] = True
    monkeypatch.setattr(clipboard_module, "is_WINDOWS", False)
    send = Mock(side_effect=AssertionError("Windows mode unavailable"))
    monkeypatch.setattr(clipboard_module, "send_unicode_text", send)
    timing = OperationTiming("recording")
    assert mixin.insert_transcribed_text("text", timing)
    send.assert_not_called()
    assert clipboard.mimeData().text() == "text"
    assert "sendinput_dispatch_start" not in timing._events


@pytest.mark.parametrize("windows,mac,enabled,fallback", [
    (True, False, True, False), (True, False, False, False), (False, False, True, False),
    (True, False, True, True), (False, True, True, False), (False, True, False, False),
])
def test_fast_paste_retains_clipboard_restore_without_blocking_waits(harness, monkeypatch, windows, mac, enabled, fallback):
    mixin, clipboard = harness
    original = QMimeData()
    original.setImageData(_make_test_image())
    clipboard.setMimeData(original)
    mixin.config.update(windows_fast_paste=enabled, windows_sendinput_text=fallback)
    monkeypatch.setattr(clipboard_module, "is_WINDOWS", windows)
    monkeypatch.setattr(clipboard_module, "is_MACOS", mac)
    monkeypatch.setattr(clipboard_module, "send_unicode_text", Mock(return_value=(0, 8)))
    waits = Mock()
    monkeypatch.setattr(clipboard_module.time, "sleep", waits)
    timing = OperationTiming("recording")
    assert mixin.insert_transcribed_text("text", timing)
    fast = (windows or mac) and enabled
    assert waits.call_count == (0 if fast else 2)
    assert mixin._clipboard_restore_timer.delay_ms == (600 if fast else 500)
    assert clipboard.mimeData().text() == "text"
    assert "paste_dispatch_end" in timing._events
    assert "paste_settle_end" in timing._events
    assert ("sendinput_fallback" in timing._events) == fallback
    mixin._clipboard_restore_timer.fire()
    assert clipboard.mimeData().hasImage()
    assert clipboard.mimeData().imageData().pixelColor(1, 1) == QColor(17, 91, 203, 177)


@pytest.mark.parametrize("fast,char", [(True, "v"), (False, "v"), (True, "c")])
def test_fast_paste_skips_only_the_alternative_library_paste_pause(harness, monkeypatch, fast, char):
    mixin, _clipboard = harness
    mixin.config.update(windows_fast_paste=fast, alt_clipboard_lib=True)
    monkeypatch.setattr(clipboard_module, "is_WINDOWS", True)
    hotkey = Mock()
    monkeypatch.setattr(clipboard_module.pyautogui, "hotkey", hotkey)
    assert ClipboardMixin._simulate_key_combination(mixin, char)
    if fast and char == "v":
        hotkey.assert_called_once_with("ctrl", char, _pause=False)
    else:
        hotkey.assert_called_once_with("ctrl", char)


@pytest.mark.parametrize("osascript_fails", [False, True])
def test_mac_fast_paste_keeps_osascript_and_removes_optional_fallback_pause(harness, monkeypatch, osascript_fails):
    mixin, _clipboard = harness
    mixin.config.update(windows_fast_paste=True, alt_clipboard_lib=True)
    monkeypatch.setattr(clipboard_module, "is_WINDOWS", False)
    monkeypatch.setattr(clipboard_module, "is_MACOS", True)
    script = Mock(side_effect=OSError("synthetic failure") if osascript_fails else None)
    hotkey = Mock()
    monkeypatch.setattr(clipboard_module.subprocess, "run", script)
    monkeypatch.setattr(clipboard_module.pyautogui, "hotkey", hotkey)
    assert ClipboardMixin._simulate_key_combination(mixin, "v")
    assert script.call_args.args[0] == ["osascript", "-e", 'tell application "System Events" to keystroke "v" using command down']
    if osascript_fails:
        hotkey.assert_called_once_with("command", "v", _pause=False)
    else:
        hotkey.assert_not_called()
