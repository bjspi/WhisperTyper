"""Regression tests for rich clipboard capture and restoration."""
from __future__ import annotations

from typing import Any

import pytest
from PyQt6.QtCore import QByteArray, QMimeData
from PyQt6.QtGui import QColor, QImage

import app.mixins.clipboard_mixin as clipboard_module
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

    def _simulate_key_combination(self, char: str) -> None:
        """Expose simulated copy/paste keys to each test without touching the OS."""
        self.on_simulated_key(char)


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
