"""The bundled colour emoji font is registered with Qt and draws flags in colour."""
from __future__ import annotations

import pytest

QtGui = pytest.importorskip("PyQt6.QtGui")

from app.ui import fonts  # noqa: E402

ENGLAND = "\U0001F3F4\U000E0067\U000E0062\U000E0065\U000E006E\U000E0067\U000E007F"


@pytest.fixture(scope="module")
def qapp():
    return QtGui.QGuiApplication.instance() or QtGui.QGuiApplication([])


def colored_pixels(text: str) -> int:
    image = QtGui.QImage(120, 60, QtGui.QImage.Format.Format_ARGB32)
    image.fill(QtGui.QColor("white"))
    painter = QtGui.QPainter(image)
    painter.setFont(QtGui.QFont(QtGui.QGuiApplication.font().family(), 32))
    painter.drawText(10, 45, text)
    painter.end()
    return sum(1 for y in range(image.height()) for x in range(image.width())
               if (lambda c: max(c.red(), c.green(), c.blue()) - min(c.red(), c.green(), c.blue()))(image.pixelColor(x, y)) > 60)


def test_subdivision_flag_is_drawn_in_colour(qapp, monkeypatch):
    monkeypatch.setattr(fonts, "is_MACOS", False)
    fonts.install_emoji_font()
    assert "Noto Color Emoji" in QtGui.QFontDatabase.applicationEmojiFontFamilies()
    assert colored_pixels(ENGLAND) > 300  # red cross on white, not a black flag outline
