"""Tray icon rendering: the application icon and the live audio-level recording icon."""
from __future__ import annotations

import os
from typing import Dict, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap

from app.core.env import is_MACOS
from app.core.paths import resource_path

_BAR_HEIGHTS = (4, 7, 10, 7, 4)
_HOT_LEVEL = 0.85


def app_icon() -> QIcon:
    """Return the platform-appropriate application icon (null if none is bundled)."""
    candidates = [resource_path("resources", "whispertyper.icns")] if is_MACOS else []
    candidates.append(resource_path("resources", "app_icon.png"))
    for icon_path in candidates:
        if os.path.exists(icon_path):
            icon = QIcon(icon_path)
            if not icon.isNull():
                return icon
    return QIcon()


def recording_level_icon(level: float, cache: Dict[Tuple[int, bool], QIcon]) -> QIcon:
    """A 16 px icon with five level bars, green or (near clipping) orange.

    There are only ~10 distinct looks (5 bar counts × hot/cool); the 80 ms refresh would
    otherwise allocate a fresh pixmap and painter ~12×/s, so finished icons are cached.
    """
    level = max(0.0, min(1.0, level))
    active_bars = max(1, min(5, int(round(level * 5))))
    hot = level >= _HOT_LEVEL
    cached = cache.get((active_bars, hot))
    if cached is not None:
        return cached

    pixmap = QPixmap(16, 16)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    painter.setPen(QPen(QColor(65, 65, 65)))
    painter.setBrush(QColor(34, 34, 34))
    painter.drawRect(0, 0, 15, 15)
    active_color = QColor(255, 120, 70) if hot else QColor(50, 205, 120)
    inactive_color = QColor(80, 80, 80)
    for index, bar_height in enumerate(_BAR_HEIGHTS):
        color = active_color if index < active_bars else inactive_color
        painter.fillRect(2 + index * 2, 13 - bar_height, 1, bar_height, color)
    painter.end()

    icon = QIcon(pixmap)
    cache[(active_bars, hot)] = icon
    return icon
