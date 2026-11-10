"""Tray context menu: a header with logo, name and hotkey, line icons, and letter accelerators.

Each entry shows its accelerator letter as a quiet hint at its right edge; pressing the letter
while the menu is open triggers the entry.

Clicks on the menu right after it opens are ignored: the menu opens upwards from the tray,
so its last entry (Quit) sits under the pointer, and a touchpad tap that lands just after
the right-click would trigger it by accident.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional

from PyQt6.QtCore import QRect, QRectF, Qt
from PyQt6.QtGui import QAction, QColor, QIcon, QKeyEvent, QMouseEvent, QPainter, QPaintEvent, QShowEvent
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QMenu, QVBoxLayout, QWidget, QWidgetAction

from app.ui.line_icons import round_corners

#: Gap between the accelerator hint and the item's right edge, and the header logo size.
_HINT_MARGIN = 14
_HEADER_ICON_SIZE = 30
#: How long after opening a click on an entry is ignored, in seconds.
CLICK_GUARD_S = 0.4


class BadgeTrayMenu(QMenu):
    """Paint each item's accelerator letter as a hint on its far right, and trigger that item
    directly when its letter is pressed while the menu is open.

    The hint is drawn in ``paintEvent`` (on top of the normal QSS render) rather than via a
    custom ``QStyle``: wrapping the widget's stylesheet style in a QProxyStyle recurses at the
    C++ level and hard-crashes on first paint. Painting over the finished menu is crash-safe and
    keeps the full QSS look (panel, rounded corners, full-width highlight, icon indent) intact.
    """

    def __init__(self, parent: Optional[QWidget], palette: Callable[[], Dict[str, str]]) -> None:
        """``palette`` returns the live theme colors at paint time."""
        super().__init__(parent)
        self._palette = palette
        self.setObjectName("tray")
        self._badge_keys: Dict[str, QAction] = {}
        self._opened_at = 0.0

    def register_badge(self, key: str, action: QAction) -> None:
        """Map a lowercase accelerator letter to the action it triggers and stamp it for paint."""
        key = key.lower()
        self._badge_keys[key] = action
        action.setData(key)  # read back in paintEvent to render the hint

    def add_header(self, icon: QIcon, title: str, status: str, version: str = "") -> QLabel:
        """Logo, name with its version and a state line above the entries (not clickable).

        Returns the state label.
        """
        header = QWidget()
        header.setObjectName("trayHeader")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(12, 8, 16, 6)
        layout.setSpacing(10)
        logo = QLabel()
        if not icon.isNull():
            logo.setPixmap(icon.pixmap(_HEADER_ICON_SIZE, _HEADER_ICON_SIZE))
        texts = QVBoxLayout()
        texts.setSpacing(0)
        name = QLabel(title)
        name.setObjectName("trayTitle")
        state = QLabel(status)
        state.setObjectName("trayStatus")
        title_row = QHBoxLayout()
        title_row.setSpacing(6)
        title_row.addWidget(name)
        if version:
            version_label = QLabel(version)
            version_label.setObjectName("trayVersion")
            title_row.addWidget(version_label, 0, Qt.AlignmentFlag.AlignBaseline)
        title_row.addStretch(1)
        texts.addLayout(title_row)
        texts.addWidget(state)
        layout.addWidget(logo)
        layout.addLayout(texts, 1)
        # Enabled, or Qt draws the logo greyed out; without a connected signal it triggers nothing.
        action = QWidgetAction(self)
        action.setDefaultWidget(header)
        self.addAction(action)
        return state

    def showEvent(self, event: Optional[QShowEvent]) -> None:  # noqa: N802 - Qt API
        """Windows 11 rounded corners and shadow; starts the click guard."""
        super().showEvent(event)
        self._opened_at = time.monotonic()
        round_corners(self)

    def mousePressEvent(self, event: Optional[QMouseEvent]) -> None:  # noqa: N802 - Qt API
        """Ignore a click on an entry that comes too soon after opening."""
        if not self._too_early(event):
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: Optional[QMouseEvent]) -> None:  # noqa: N802 - Qt API
        """Ignore a release on an entry that comes too soon after opening."""
        if not self._too_early(event):
            super().mouseReleaseEvent(event)

    def _too_early(self, event: Optional[QMouseEvent]) -> bool:
        """A click inside the menu within the guard time; outside clicks still close it."""
        if event is None or not self.rect().contains(event.position().toPoint()):
            return False
        return time.monotonic() - self._opened_at < CLICK_GUARD_S

    def keyPressEvent(self, event: Optional[QKeyEvent]) -> None:  # noqa: N802 - Qt API
        """Trigger the action whose accelerator letter was pressed; a submenu's letter opens it."""
        if event is not None:
            action = self._badge_keys.get((event.text() or "").lower())
            if action is not None and action.isEnabled() and action.menu() is not None:
                self.setActiveAction(action)
                return
            if action is not None and action.isEnabled():
                self.close()
                action.trigger()
                return
        super().keyPressEvent(event)

    def paintEvent(self, event: Optional[QPaintEvent]) -> None:  # noqa: N802 - Qt API
        """Paint the accelerator hints over the regular menu render.

        The update state shows in the entry's text and accent icon (see the tray controller).
        """
        super().paintEvent(event)
        colors = self._palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for action in self.actions():
            letter = action.data()
            if action.isSeparator() or not action.isVisible() or not isinstance(letter, str) or not letter:
                continue
            self._paint_hint(painter, self.actionGeometry(action), letter, action.isEnabled(), colors)
        painter.end()

    def _paint_hint(self, painter: QPainter, rect: QRect, letter: str, enabled: bool,
                    colors: Dict[str, str]) -> None:
        """Draw the accelerator letter right-aligned in a muted colour, like a shortcut hint."""
        box = QRectF(rect.left(), rect.top(), rect.width() - _HINT_MARGIN, rect.height())
        color = QColor(colors["muted"])
        if not enabled:
            color.setAlpha(110)
        painter.setPen(color)
        painter.setFont(self.font())
        painter.drawText(box, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, letter.upper())
