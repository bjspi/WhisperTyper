"""Regression tests for rich clipboard capture and restoration."""
from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest
from PyQt6.QtCore import QByteArray, QMimeData
from PyQt6.QtGui import QColor, QImage

import app.controllers.text_output as clipboard_module
import app.services.clipboard as snapshot_module
from app.controllers.text_output import TextOutput
from app.core.timing import OperationTiming
from app.services.clipboard import ClipboardSnapshot


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


class ClipboardHarness(TextOutput):
    def __init__(self) -> None:
        """Initialize the clipboard state and controllable restore timer."""
        self.config = {"restore_clipboard": True}
        self.notify = Mock()
        super().__init__(self.config, lambda key, **_kwargs: key, self.notify, lambda _permission: None)
        self.on_simulated_key = lambda _char: None
        self._restore_timer = FakeTimer(self.restore_clipboard_now)

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

    monkeypatch.setattr(snapshot_module, "QApplication", FakeApplication)
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
    output, clipboard = harness
    original_mime = QMimeData()
    original_mime.setHtml("<p><b>rich text</b></p>")
    original_mime.setData("application/x-whispertyper-test", QByteArray(b"\x00\xffblob\x00"))
    original_mime.setImageData(_make_test_image())
    clipboard.setMimeData(original_mime)

    snapshot = ClipboardSnapshot.capture()
    clipboard.setMimeData(QMimeData())
    clipboard.mimeData().setText("temporary transcription")
    snapshot.restore()

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
    _output, clipboard = harness
    image_mime = QMimeData()
    image_mime.setImageData(_make_test_image())
    clipboard.setMimeData(image_mime)

    assert bytes(image_mime.data("application/x-qt-image")) == b""

    snapshot = ClipboardSnapshot.capture()
    assert isinstance(snapshot.image, QImage)

    clipboard.clear()
    snapshot.restore()
    restored_image: Any = clipboard.mimeData().imageData()
    assert isinstance(restored_image, QImage)
    assert restored_image.pixelColor(2, 1) == QColor(17, 91, 203, 177)


def test_empty_clipboard_roundtrip_stays_empty(
    harness: tuple[ClipboardHarness, FakeClipboard],
) -> None:
    _output, clipboard = harness

    snapshot = ClipboardSnapshot.capture()
    temporary = QMimeData()
    temporary.setText("temporary transcription")
    clipboard.setMimeData(temporary)
    snapshot.restore()

    assert clipboard.mimeData().formats() == []


def test_insert_path_keeps_text_until_delayed_restore_even_when_paste_simulation_fails(
    harness: tuple[ClipboardHarness, FakeClipboard],
) -> None:
    output, clipboard = harness
    original_mime = QMimeData()
    original_mime.setImageData(_make_test_image())
    clipboard.setMimeData(original_mime)

    def fail_on_paste(char: str) -> None:
        assert char == "v"
        assert clipboard.mimeData().text() == "temporary transcription"
        raise RuntimeError("synthetic paste failure")

    output.on_simulated_key = fail_on_paste
    output.insert("temporary transcription")

    assert clipboard.mimeData().text() == "temporary transcription"
    assert output._restore_timer.delay_ms == 500
    assert output.restore_pending

    output._restore_timer.fire()
    restored_image: Any = clipboard.mimeData().imageData()
    assert isinstance(restored_image, QImage)
    assert restored_image.pixelColor(1, 1) == QColor(17, 91, 203, 177)


def test_selected_text_path_restores_non_text_clipboard(
    harness: tuple[ClipboardHarness, FakeClipboard],
) -> None:
    output, clipboard = harness
    original_mime = QMimeData()
    original_mime.setData("application/x-whispertyper-test", QByteArray(b"original blob"))
    original_mime.setImageData(_make_test_image())
    clipboard.setMimeData(original_mime)

    def copy_selection(char: str) -> None:
        assert char == "c"
        selected_mime = QMimeData()
        selected_mime.setText("  selected for rephrasing  ")
        clipboard.setMimeData(selected_mime)

    output.on_simulated_key = copy_selection
    selected = output.get_selected_text()

    assert selected == "selected for rephrasing"
    assert bytes(clipboard.mimeData().data("application/x-whispertyper-test")) == b"original blob"
    assert clipboard.mimeData().hasImage()


def test_commit_timings_separate_paste_dispatch_from_existing_waits(harness, monkeypatch):
    output, clipboard = harness
    clock = [1_000_000_000]
    monkeypatch.setattr("app.core.timing.time.perf_counter_ns", lambda: clock[0])
    monkeypatch.setattr(clipboard_module.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + int(seconds * 1e9)))
    timing = OperationTiming("recording")
    assert output.insert("private transcript", timing=timing)
    events = timing._events
    assert events["paste_dispatch_start"] - events["clipboard_write_end"] == 100_000_000
    assert events["paste_settle_end"] - events["paste_dispatch_end"] == 100_000_000
    assert events["text_commit_end"] == events["paste_dispatch_end"]
    assert events["text_commit_end"] < events["paste_settle_end"]
    assert "text_commit_failed" not in events
    assert clipboard.mimeData().text() == "private transcript"
    assert output._restore_timer.active


@pytest.mark.parametrize("failure", ["copy", "paste_exception", "paste_false"])
def test_failed_commit_does_not_invent_success_and_preserves_restore(harness, monkeypatch, failure):
    output, clipboard = harness
    original = QMimeData()
    original.setText("original clipboard")
    clipboard.setMimeData(original)

    def fail(*_args):
        raise RuntimeError("synthetic failure")

    if failure == "copy":
        monkeypatch.setattr(clipboard_module.copykitten, "copy", fail)
    elif failure == "paste_exception":
        output.on_simulated_key = fail
    else:
        monkeypatch.setattr(output, "_simulate_key_combination", lambda _char: False)
    timing = OperationTiming("recording")
    assert not output.insert("private transcript", timing=timing)
    assert "text_commit_failed" in timing._events
    assert "text_commit_end" not in timing._events
    assert "paste_dispatch_end" not in timing._events
    assert output.restore_pending


def test_empty_text_has_no_commit_milestones(harness):
    output, _clipboard = harness
    timing = OperationTiming("recording")
    assert output.insert("", timing=timing)
    assert "text_commit_start" not in timing._events


def test_direct_input_preserves_rich_clipboard_and_has_no_paste_waits(harness, monkeypatch):
    output, clipboard = harness
    output.config["windows_sendinput_text"] = True
    monkeypatch.setattr(clipboard_module, "is_WINDOWS", True)
    send = Mock(return_value=(8, 8))
    monkeypatch.setattr(clipboard_module, "send_unicode_text", send)
    original = QMimeData()
    original.setImageData(_make_test_image())
    clipboard.setMimeData(original)
    copy = Mock(side_effect=AssertionError("Direct input must not touch clipboard"))
    monkeypatch.setattr(clipboard_module.copykitten, "copy", copy)
    monkeypatch.setattr(output, "_capture_clipboard_state", copy)
    monkeypatch.setattr(clipboard_module.time, "sleep", copy)
    monkeypatch.setattr(output, "_simulate_key_combination", copy)
    timing = OperationTiming("recording")
    assert output.insert("ä🦄文", timing)
    send.assert_called_once_with("ä🦄文")
    assert clipboard.mimeData() is original
    assert not output._restore_timer.active
    assert "sendinput_dispatch_end" in timing._events
    assert "text_commit_end" in timing._events
    assert "paste_dispatch_start" not in timing._events
    assert "clipboard_snapshot_start" not in timing._events


@pytest.mark.parametrize("rejection", [(0, 8), OSError("modifier held"), ValueError("control character")])
def test_direct_rejection_can_fallback_to_existing_paste(harness, monkeypatch, rejection):
    output, clipboard = harness
    output.config.update(windows_sendinput_text=True, windows_sendinput_fallback=True)
    monkeypatch.setattr(clipboard_module, "is_WINDOWS", True)
    send = Mock(side_effect=rejection) if isinstance(rejection, Exception) else Mock(return_value=rejection)
    monkeypatch.setattr(clipboard_module, "send_unicode_text", send)
    paste = Mock()
    output.on_simulated_key = paste
    timing = OperationTiming("recording")
    assert output.insert("text", timing)
    paste.assert_called_once_with("v")
    assert clipboard.mimeData().text() == "text"
    assert output._restore_timer.active
    assert "sendinput_fallback" in timing._events
    assert "paste_dispatch_end" in timing._events
    assert "text_commit_end" in timing._events
    assert "text_commit_failed" not in timing._events


@pytest.mark.parametrize("result,fallback", [((1, 8), True), ((3, 8), True), ((0, 8), False), (RuntimeError(), True)])
def test_partial_or_unknown_failure_never_duplicates_text(harness, monkeypatch, result, fallback):
    output, clipboard = harness
    original = clipboard.mimeData()
    output.config.update(windows_sendinput_text=True, windows_sendinput_fallback=fallback)
    monkeypatch.setattr(clipboard_module, "is_WINDOWS", True)
    send = Mock(side_effect=result) if isinstance(result, Exception) else Mock(return_value=result)
    monkeypatch.setattr(clipboard_module, "send_unicode_text", send)
    paste = Mock()
    output.on_simulated_key = paste
    timing = OperationTiming("recording")
    assert not output.insert("text", timing)
    paste.assert_not_called()
    output.notify.assert_called_once()
    assert clipboard.mimeData() is original
    assert not output._restore_timer.active
    assert "text_commit_failed" in timing._events
    assert "text_commit_end" not in timing._events
    assert "sendinput_fallback" not in timing._events


def test_saved_windows_option_has_no_effect_off_windows(harness, monkeypatch):
    output, clipboard = harness
    output.config["windows_sendinput_text"] = True
    monkeypatch.setattr(clipboard_module, "is_WINDOWS", False)
    send = Mock(side_effect=AssertionError("Windows mode unavailable"))
    monkeypatch.setattr(clipboard_module, "send_unicode_text", send)
    timing = OperationTiming("recording")
    assert output.insert("text", timing)
    send.assert_not_called()
    assert clipboard.mimeData().text() == "text"
    assert "sendinput_dispatch_start" not in timing._events


@pytest.mark.parametrize("fast", [False, True])
def test_fast_paste_skips_waits_but_keeps_a_later_clipboard_restore(harness, monkeypatch, fast):
    output, clipboard = harness
    original = QMimeData()
    original.setImageData(_make_test_image())
    clipboard.setMimeData(original)
    output.config["fast_paste"] = fast
    monkeypatch.setattr(clipboard_module, "is_WINDOWS", True)
    waits = Mock()
    monkeypatch.setattr(clipboard_module.time, "sleep", waits)
    timing = OperationTiming("recording")
    assert output.insert("text", timing)
    assert waits.call_count == (0 if fast else 2)
    assert output._restore_timer.delay_ms == (600 if fast else 500)
    assert clipboard.mimeData().text() == "text"
    assert "paste_settle_end" in timing._events
    output._restore_timer.fire()
    assert clipboard.mimeData().imageData().pixelColor(1, 1) == QColor(17, 91, 203, 177)
