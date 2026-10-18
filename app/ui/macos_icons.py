"""macOS menu-bar icons: a crisp monochrome WhisperTyper mark and its recording variant."""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QApplication

from app.core.env import is_MACOS

_WHITE = QColor(255, 255, 255)


def _status_symbol(size: int, color: QColor) -> QPixmap:
    """Draw a simplified tray-only WhisperTyper mark that stays legible in the macOS menu bar."""
    if size <= 0:
        return QPixmap()
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(color)
    design_size = 24.0
    scale = size / design_size
    bar_height = 3.2 * scale
    radius = bar_height / 2.0
    left_bars = [(4.2, 1.8, 5.8), (2.8, 6.3, 7.8), (1.5, 10.8, 9.8), (3.2, 15.4, 8.2), (5.0, 19.8, 6.4)]
    for x, y, width in left_bars:
        painter.drawRoundedRect(QRectF(x * scale, y * scale, width * scale, bar_height), radius, radius)
        mirrored_x = design_size - x - width
        painter.drawRoundedRect(QRectF(mirrored_x * scale, y * scale, width * scale, bar_height), radius, radius)
    painter.end()
    return pixmap


def _status_pixmap(size: int, color: Optional[QColor] = None) -> QPixmap:
    """A scaled menu-bar status pixmap, optionally recolored (empty off macOS)."""
    if not is_MACOS:
        return QPixmap()
    return _status_symbol(size, color if color is not None else QColor(0, 0, 0))


def _device_pixel_ratio() -> float:
    """Device pixel ratio used for crisp menu-bar icons."""
    screen = QApplication.primaryScreen()
    if screen is not None:
        try:
            return max(1.0, float(screen.devicePixelRatio()))
        except Exception:
            pass
    return 2.0 if is_MACOS else 1.0


def tray_icon(size: int = 18) -> QIcon:
    """The idle menu-bar icon as a white monochrome symbol."""
    scale = _device_pixel_ratio()
    pixmap = _status_pixmap(max(18, int(round(size * scale))), _WHITE)
    if pixmap.isNull():
        return QIcon()
    pixmap.setDevicePixelRatio(scale)
    return QIcon(pixmap)


def recording_tray_icon(level: float, brand_icon: QIcon) -> QIcon:
    """The menu-bar icon while recording: the mark plus a level-sized indicator dot."""
    scale = _device_pixel_ratio()
    base_size = max(20, int(round(20 * scale)))
    base_pixmap = _status_pixmap(base_size, _WHITE)
    if base_pixmap.isNull() and not brand_icon.isNull():
        base_pixmap = brand_icon.pixmap(base_size, base_size)
    if base_pixmap.isNull():
        return QIcon()

    canvas_size = max(24, int(round(24 * scale)))
    pixmap = QPixmap(canvas_size, canvas_size)
    pixmap.fill(Qt.GlobalColor.transparent)
    pixmap.setDevicePixelRatio(scale)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    icon_offset = int(round(2 * scale))
    painter.drawPixmap(icon_offset, icon_offset, base_pixmap)

    indicator_size = 5 + max(0, min(3, int(round(level * 3))))
    indicator_color = QColor(16, 199, 222) if level < 0.82 else QColor(18, 124, 243)
    indicator_size_px = max(4, int(round(indicator_size * scale)))
    indicator_offset = int(round(2 * scale))
    painter.setPen(QPen(QColor(255, 255, 255, 230), 1))
    painter.setBrush(indicator_color)
    corner = canvas_size - indicator_size_px - indicator_offset
    painter.drawEllipse(corner, corner, indicator_size_px, indicator_size_px)
    painter.end()
    return QIcon(pixmap)
