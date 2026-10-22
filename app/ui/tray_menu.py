"""Tray context menu with letter-accelerator badges and an "update available" dot.

In the modern design the menu gets a header with logo, name and hotkey,
line icons, and the accelerator letters as quiet hints instead of boxed badges.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

from PyQt6.QtCore import QPointF, QRect, QRectF, Qt
from PyQt6.QtGui import QAction, QColor, QIcon, QKeyEvent, QPainter, QPaintEvent, QPen, QShowEvent
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QMenu, QStyle, QVBoxLayout, QWidget, QWidgetAction

from app.ui.line_icons import round_corners

_BADGE_BOX = 18   # accelerator badge square size in px
_BADGE_MARGIN = 8  # gap between the badge and the item's right edge
_UPDATE_DOT_COLOR = QColor(46, 204, 113)
#: Modern design: gap between the accelerator hint and the item's right edge, and the header logo size.
_HINT_MARGIN = 14
_HEADER_ICON_SIZE = 30


class BadgeTrayMenu(QMenu):
    """Paint a rounded-square accelerator badge on the far right of each item, and trigger
    that item directly when its letter is pressed while the menu is open.

    The badge is drawn in ``paintEvent`` (on top of the normal QSS render) rather than via a
    custom ``QStyle``: wrapping the widget's stylesheet style in a QProxyStyle recurses at the
    C++ level and hard-crashes on first paint. Painting over the finished menu is crash-safe and
    keeps the full QSS look (panel, rounded corners, full-width highlight, icon indent) intact.
    """

    def __init__(self, parent: Optional[QWidget], palette: Callable[[], Dict[str, str]], *,
                 modern: bool = False) -> None:
        """``palette`` returns the live theme colors at paint time; ``modern`` selects the modern design."""
        super().__init__(parent)
        self._palette = palette
        self._modern = modern
        if modern:
            self.setObjectName("tray")
        self._badge_keys: Dict[str, QAction] = {}
        #: The menu entry that shows the green dot while ``update_available`` is set.
        self.update_action: Optional[QAction] = None
        self.update_available = False

    def register_badge(self, key: str, action: QAction) -> None:
        """Map a lowercase accelerator letter to the action it triggers and stamp it for paint."""
        key = key.lower()
        self._badge_keys[key] = action
        action.setData(key)  # read back in paintEvent to render the badge

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
        """Modern design: Windows 11 rounded corners and shadow."""
        super().showEvent(event)
        if self._modern:
            round_corners(self)

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
            if self._modern:
                # The update state shows in the entry's text and accent icon instead of a dot.
                self._paint_hint(painter, self.actionGeometry(action), letter, action.isEnabled(), colors)
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
