"""Tray context menu with letter-accelerator badges and an "update available" dot."""
from __future__ import annotations

from typing import Callable, Dict, Optional

from PyQt6.QtCore import QPointF, QRect, QRectF, Qt
from PyQt6.QtGui import QAction, QColor, QKeyEvent, QPainter, QPaintEvent, QPen
from PyQt6.QtWidgets import QMenu, QStyle, QWidget

_BADGE_BOX = 18   # accelerator badge square size in px
_BADGE_MARGIN = 8  # gap between the badge and the item's right edge
_UPDATE_DOT_COLOR = QColor(46, 204, 113)


class BadgeTrayMenu(QMenu):
    """Paint a rounded-square accelerator badge on the far right of each item, and trigger
    that item directly when its letter is pressed while the menu is open.

    The badge is drawn in ``paintEvent`` (on top of the normal QSS render) rather than via a
    custom ``QStyle``: wrapping the widget's stylesheet style in a QProxyStyle recurses at the
    C++ level and hard-crashes on first paint. Painting over the finished menu is crash-safe and
    keeps the full QSS look (panel, rounded corners, full-width highlight, icon indent) intact.
    """

    def __init__(self, parent: Optional[QWidget], palette: Callable[[], Dict[str, str]]) -> None:
        """``palette`` returns the live theme colors at paint time."""
        super().__init__(parent)
        self._palette = palette
        self._badge_keys: Dict[str, QAction] = {}
        #: The menu entry that shows the green dot while ``update_available`` is set.
        self.update_action: Optional[QAction] = None
        self.update_available = False

    def register_badge(self, key: str, action: QAction) -> None:
        """Map a lowercase accelerator letter to the action it triggers and stamp it for paint."""
        key = key.lower()
        self._badge_keys[key] = action
        action.setData(key)  # read back in paintEvent to render the badge

    def keyPressEvent(self, event: Optional[QKeyEvent]) -> None:  # noqa: N802 - Qt API
        """Trigger the action whose badge letter was pressed."""
        if event is not None:
            action = self._badge_keys.get((event.text() or "").lower())
            if action is not None and action.isEnabled():
                self.close()
                action.trigger()
                return
        super().keyPressEvent(event)

    def paintEvent(self, event: Optional[QPaintEvent]) -> None:  # noqa: N802 - Qt API
        """Paint badges (and the update dot) over the regular menu render."""
        super().paintEvent(event)
        colors = self._palette()
        active = self.activeAction()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for action in self.actions():
            letter = action.data()
            if action.isSeparator() or not isinstance(letter, str) or not letter:
                continue
            self._paint_badge(painter, self.actionGeometry(action), letter,
                              action.isEnabled(), action is active, colors)
            # Raised like a superscript just after the label text (kept off the icon).
            if self.update_available and action is self.update_action:
                self._paint_update_dot(painter, self.actionGeometry(action), action.text())
        painter.end()

    def _paint_badge(self, painter: QPainter, rect: QRect, letter: str,
                     enabled: bool, selected: bool, colors: Dict[str, str]) -> None:
        """Draw one rounded accelerator badge."""
        x = rect.right() - _BADGE_MARGIN - _BADGE_BOX
        y = rect.center().y() - _BADGE_BOX // 2
        box = QRectF(x, y, _BADGE_BOX, _BADGE_BOX)
        if selected and enabled:
            border = QColor(colors["on_accent"])
            border.setAlpha(150)
            fill = QColor(colors["on_accent"])
            fill.setAlpha(30)
            fg = QColor(colors["on_accent"])
        else:
            border = QColor(colors["border"])
            fill = QColor(colors["panel2"])
            fg = QColor(colors["muted"])
        if not enabled:
            border.setAlpha(70)
            fg.setAlpha(110)
        painter.setPen(QPen(border, 1))
        painter.setBrush(fill)
        painter.drawRoundedRect(box, 4.0, 4.0)
        font = painter.font()
        font.setBold(True)
        font.setPointSizeF(max(7.5, font.pointSizeF() * 0.85))
        painter.setFont(font)
        painter.setPen(fg)
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, letter.upper())

    def _paint_update_dot(self, painter: QPainter, rect: QRect, text: str) -> None:
        """Paint a small raised green dot just after the item's text (superscript-style)."""
        fm = self.fontMetrics()
        # Text starts after the QSS item left-padding (15px) + the reserved icon column.
        style = self.style()
        icon_extent = style.pixelMetric(QStyle.PixelMetric.PM_SmallIconSize) if style is not None else 16
        text_left = rect.left() + 15 + icon_extent + 5
        cx = text_left + fm.horizontalAdvance(text) + 8
        cy = rect.center().y() - max(3, fm.ascent() // 3)  # lifted above the baseline
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_UPDATE_DOT_COLOR)
        painter.drawEllipse(QPointF(float(cx), float(cy)), 3.5, 3.5)
