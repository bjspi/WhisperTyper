"""Standalone end-to-end check for image/blob clipboard restoration.

Run from the repository root with::

    python tests/manual_clipboard_image_restore.py

The script creates its own deterministic image and binary clipboard format, temporarily
replaces them with text, and verifies the app's snapshot/restore code. Whatever was in
the system clipboard before the test is restored in a ``finally`` block.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import copykitten
from PyQt6.QtCore import QByteArray, QMimeData
from PyQt6.QtGui import QColor, QImage
from PyQt6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.mixins.clipboard_mixin import ClipboardMixin  # noqa: E402


def main() -> int:
    app = QApplication.instance() or QApplication([])
    clipboard = QApplication.clipboard()
    mixin = ClipboardMixin()
    mixin._pending_clipboard_restore_state = None
    original_state = mixin._capture_clipboard_state()

    try:
        # First reproduce the real Windows path with a native image written and read
        # through copykitten, while WhisperTyper temporarily writes transcription text.
        native_rgba = bytes((23, 117, 211, 149)) * (7 * 5)
        copykitten.copy_image(native_rgba, 7, 5)
        app.processEvents()
        native_image_state = mixin._capture_clipboard_state()
        native_snapshot = native_image_state["mime_snapshot"]
        original_native_formats = {
            entry["format"]: entry["data"]
            for entry in native_snapshot["formats"]
            if entry["data"]
        }
        copykitten.copy("temporary transcription")
        app.processEvents()
        mixin._restore_clipboard_state(native_image_state)

        restored_native_mime = clipboard.mimeData()
        restored_native_image: Any = restored_native_mime.imageData()
        assert isinstance(restored_native_image, QImage), "native image was not restored"
        assert (restored_native_image.width(), restored_native_image.height()) == (7, 5), (
            "native image size differs"
        )
        assert restored_native_image.pixelColor(3, 2) == QColor(23, 117, 211, 149), (
            "native image pixels differ"
        )
        for mime_format, expected_bytes in original_native_formats.items():
            assert restored_native_mime.data(mime_format).data() == expected_bytes, (
                f"native clipboard format {mime_format!r} differs"
            )

        # Then cover an arbitrary binary MIME object alongside an image. This catches
        # formats that have no copykitten text or image representation of their own.
        expected_image = QImage(7, 5, QImage.Format.Format_ARGB32)
        expected_color = QColor(23, 117, 211, 149)
        expected_image.fill(expected_color)
        expected_blob = b"\x00WhisperTyper\xffclipboard-test\x00"

        generated_mime = QMimeData()
        generated_mime.setImageData(expected_image)
        generated_mime.setData("application/x-whispertyper-test", QByteArray(expected_blob))
        clipboard.setMimeData(generated_mime)
        app.processEvents()

        generated_state = mixin._capture_clipboard_state()
        # Use the same text-writing library as the real insertion path.
        copykitten.copy("temporary transcription")
        app.processEvents()

        mixin._restore_clipboard_state(generated_state)
        restored_mime = clipboard.mimeData()
        restored_image: Any = restored_mime.imageData()

        assert isinstance(restored_image, QImage), "restored clipboard contains no QImage"
        assert restored_image.size() == expected_image.size(), "restored image size differs"
        assert restored_image.pixelColor(3, 2) == expected_color, "restored image pixels differ"
        assert bytes(restored_mime.data("application/x-whispertyper-test")) == expected_blob, (
            "restored binary clipboard payload differs"
        )
        print("PASS: native image pixels and binary clipboard payload were restored exactly.")
        return 0
    finally:
        mixin._restore_clipboard_state(original_state)
        app.processEvents()


if __name__ == "__main__":
    raise SystemExit(main())
