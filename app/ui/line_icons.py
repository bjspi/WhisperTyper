"""Tinted line icons (``resources/icons``) and rounded Windows 11 popup menus for the modern design."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from PyQt6.QtCore import QByteArray, QRectF, Qt
from PyQt6.QtGui import QIcon, QPainter, QPixmap, QShowEvent
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QMenu

from app.core.env import is_WINDOWS
from app.core.paths import resource_path

#: The stroke colour the SVG files are drawn with; replaced by the tint.
_TINT_PLACEHOLDER = "#000000"
_ICON_SIZES = (16, 18, 20, 24, 32)
#: DWM attribute and value for small rounded window corners with the native shadow (Windows 11).
_DWMWA_WINDOW_CORNER_PREFERENCE = 33
_DWMWCP_ROUNDSMALL = 3


@lru_cache(maxsize=None)
def line_icon(name: str, color: str, selected_color: Optional[str] = None) -> QIcon:
    """The icon ``resources/icons/<name>.svg`` in ``color``; ``selected_color`` tints the selected state."""
    icon = QIcon()
    try:
        with open(resource_path("resources", "icons", f"{name}.svg"), encoding="utf-8") as handle:
            source = handle.read()
    except OSError:
        return icon
    for mode, tint in ((QIcon.Mode.Normal, color), (QIcon.Mode.Selected, selected_color or color)):
        renderer = QSvgRenderer(QByteArray(source.replace(_TINT_PLACEHOLDER, tint).encode()))
        for size in _ICON_SIZES:
            pixmap = QPixmap(size * 2, size * 2)  # sharp on high-DPI screens
            pixmap.setDevicePixelRatio(2)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            renderer.render(painter, QRectF(0, 0, size, size))
            painter.end()
            icon.addPixmap(pixmap, mode)
    return icon


def round_corners(menu: QMenu) -> None:
    """Ask Windows 11 for rounded corners and the native shadow; other systems ignore it.

    The window stays opaque, so its text keeps ClearType rendering.
    """
    if not is_WINDOWS:
        return
    import ctypes

    preference = ctypes.c_int(_DWMWCP_ROUNDSMALL)
    try:
        ctypes.windll.dwmapi.DwmSetWindowAttribute(  # type: ignore[attr-defined]
            int(menu.winId()), _DWMWA_WINDOW_CORNER_PREFERENCE, ctypes.byref(preference),
            ctypes.sizeof(preference))
    except (AttributeError, OSError):
        pass


class RoundedMenu(QMenu):
    """A QMenu with Windows 11 rounded corners and shadow."""

    def showEvent(self, event: Optional[QShowEvent]) -> None:  # noqa: N802 - Qt API
        """Round the popup each time it opens (Qt may recreate the native window)."""
        super().showEvent(event)
        round_corners(self)
