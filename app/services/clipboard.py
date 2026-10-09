"""Complete clipboard snapshots, so temporary copy/paste never destroys the user's clipboard."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import copykitten
from PyQt6.QtCore import QByteArray, QMimeData
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication

MimeFormats = tuple[tuple[str, bytes], ...]


@dataclass(frozen=True)
class ClipboardSnapshot:
    """The clipboard's native formats (incl. images and custom binary data) plus its text."""

    #: Plain text, or None when it could not be read (restoring then clears the clipboard).
    text: Optional[str]
    #: Every native MIME format, or None when the Qt clipboard was unavailable.
    formats: Optional[MimeFormats] = None
    #: Detached pixels: Qt often reports ``application/x-qt-image`` with zero raw bytes.
    image: Optional[QImage] = None

    @classmethod
    def capture(cls) -> ClipboardSnapshot:
        """Snapshot the current clipboard; Qt exposes native formats on Windows and macOS."""
        formats, image = _capture_mime()
        try:
            text: Optional[str] = copykitten.paste()
        except Exception:
            text = None
            if formats is None:
                logging.warning("Could not read initial clipboard text state for restoration.")
        return cls(text, formats, image)

    def restore(self) -> None:
        """Put the snapshot back, falling back to its plain text if the native restore fails."""
        if self.formats is not None:
            if _restore_mime(self.formats, self.image):
                logging.debug("Clipboard content restored.")
                return
            logging.warning("Falling back to plain-text clipboard restoration.")
        if self.text is not None:
            copykitten.copy(self.text)
        else:
            copykitten.clear()
        logging.debug("Clipboard content restored.")


def _capture_mime() -> tuple[Optional[MimeFormats], Optional[QImage]]:
    """Copy every MIME format's bytes and, separately, the semantic image data."""
    try:
        clipboard = QApplication.clipboard()
        mime_data = clipboard.mimeData() if clipboard else None
    except Exception as e:
        logging.warning(f"Could not access Qt clipboard for snapshot: {e}")
        return None, None
    if mime_data is None:
        return (), None

    formats = []
    for mime_format in mime_data.formats():
        try:
            formats.append((mime_format, mime_data.data(mime_format).data()))
        except Exception as e:
            logging.warning(f"Could not snapshot clipboard format '{mime_format}': {e}")

    image: Optional[QImage] = None
    if mime_data.hasImage():
        try:
            image = _detached_image(mime_data.imageData())
        except Exception as e:
            logging.warning(f"Could not snapshot clipboard image data: {e}")
    return tuple(formats), image


def _detached_image(image_data: Any) -> Optional[QImage]:
    """A QImage copy that outlives the clipboard's own data."""
    if isinstance(image_data, QImage):
        return image_data.copy()
    if isinstance(image_data, QPixmap):
        return image_data.toImage().copy()
    logging.warning("Clipboard advertised image data in unsupported Qt type %s.", type(image_data).__name__)
    return None


def _restore_mime(formats: MimeFormats, image: Optional[QImage]) -> bool:
    """Set the captured formats again; Qt takes ownership of the parentless QMimeData.

    Raw formats go first and the semantic image last, because setting an empty
    ``application/x-qt-image`` byte array after ``setImageData`` would erase the pixels.
    """
    try:
        clipboard = QApplication.clipboard()
    except Exception as e:
        logging.warning(f"Could not access Qt clipboard for restoration: {e}")
        return False
    if clipboard is None:
        return False
    try:
        if not formats and image is None:
            clipboard.clear()
        else:
            restored = QMimeData()
            for mime_format, data in formats:
                if mime_format:
                    restored.setData(mime_format, QByteArray(bytes(data)))
            if image is not None:
                restored.setImageData(image.copy())
            clipboard.setMimeData(restored)
        QApplication.processEvents()
        return True
    except Exception as e:
        logging.warning(f"Failed to restore Qt clipboard MIME snapshot: {e}")
        return False
