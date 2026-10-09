"""macOS menu-bar icons: a crisp monochrome WhisperTyper mark and its recording variant."""
from __future__ import annotations

import os
from typing import Optional

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QApplication

from app.core.env import is_MACOS
from app.core.paths import resource_path

_WHITE = QColor(255, 255, 255)


def _status_icon() -> QIcon:
    """Return the bundled menu-bar status icon, preferring a template image."""
    if not is_MACOS:
        return QIcon()
    for icon_path in (resource_path("resources", "whispertyperStatusTemplate.png"),
                      resource_path("resources", "whispertyper-status.png")):
        if os.path.exists(icon_path):
            icon = QIcon(icon_path)
            if not icon.isNull():
                return icon
    return QIcon()


def _tint(pixmap: QPixmap, color: QColor) -> QPixmap:
    """Tint a pixmap while preserving its original alpha mask."""
    if pixmap.isNull():
        return QPixmap()
    tinted = QPixmap(pixmap.size())
    tinted.fill(Qt.GlobalColor.transparent)
    painter = QPainter(tinted)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.drawPixmap(0, 0, pixmap)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    painter.fillRect(tinted.rect(), color)
    painter.end()
    return tinted


def _crop_to_visible_bounds(pixmap: QPixmap) -> QPixmap:
    """Trim transparent padding around a pixmap so status icons render at full size."""
    if pixmap.isNull():
        return QPixmap()
    image = pixmap.toImage()
    min_x, min_y, max_x, max_y = image.width(), image.height(), -1, -1
    for y in range(image.height()):
        for x in range(image.width()):
            if image.pixelColor(x, y).alpha() > 0:
                min_x, min_y = min(min_x, x), min(min_y, y)
                max_x, max_y = max(max_x, x), max(max_y, y)
    if max_x < min_x or max_y < min_y:
        return pixmap
    padding = max(1, int(round(max(max_x - min_x + 1, max_y - min_y + 1) * 0.05)))
    min_x, min_y = max(0, min_x - padding), max(0, min_y - padding)
    max_x, max_y = min(image.width() - 1, max_x + padding), min(image.height() - 1, max_y + padding)
    return QPixmap.fromImage(image.copy(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1))


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
    """A scaled menu-bar status pixmap, optionally recolored."""
    if is_MACOS:
        symbol = _status_symbol(size, color if color is not None else QColor(0, 0, 0))
        if not symbol.isNull():
            return symbol
    icon = _status_icon()
    if icon.isNull():
        return QPixmap()
    source_size = max(size * 8, 128)
    pixmap = _crop_to_visible_bounds(icon.pixmap(source_size, source_size))
    if not pixmap.isNull():
        pixmap = pixmap.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
    if pixmap.isNull() or color is None:
        return pixmap
    return _tint(pixmap, color)


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
