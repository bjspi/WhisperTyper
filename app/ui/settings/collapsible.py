"""A titled settings section whose content is shown or hidden by clicking its header."""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtGui import QEnterEvent, QMouseEvent
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from app.ui.theme import set_style_state

_COLLAPSED, _EXPANDED = "▸", "▾"


class _SectionHeader(QWidget):
    """Flat, full-width header row: chevron and title, then a thin line to the right edge."""

    clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Build the row; the whole row is the click target."""
        super().__init__(parent)
        self.setObjectName("collapsible_header")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.title = QLabel(self)
        self.title.setObjectName("collapsible_title")
        line = QFrame(self)
        line.setObjectName("collapsible_line")
        line.setFrameShape(QFrame.Shape.HLine)
        line.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 6, 2, 6)
        layout.setSpacing(10)
        layout.addWidget(self.title)
        layout.addWidget(line, 1)

    def mouseReleaseEvent(self, event: Optional[QMouseEvent]) -> None:  # noqa: N802 - Qt API
        """A left click anywhere on the row toggles the section."""
        if event is not None and event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)

    def enterEvent(self, event: Optional[QEnterEvent]) -> None:  # noqa: N802 - Qt API
        """Hover highlight through a style state (the theme colours it)."""
        set_style_state(self.title, "hovered", True)
        super().enterEvent(event)

    def leaveEvent(self, event: Optional[QEvent]) -> None:  # noqa: N802 - Qt API
        """End the hover highlight."""
        set_style_state(self.title, "hovered", False)
        super().leaveEvent(event)


class CollapsibleSection(QWidget):
    """Section header with a chevron; the content below it is visible only while expanded."""

    toggled = pyqtSignal(bool)

    def __init__(self, parent: Optional[QWidget] = None, expanded: bool = False) -> None:
        """Build the header and an empty content area (fill it through ``content_layout``)."""
        super().__init__(parent)
        self._title = ""
        self._expanded = expanded
        self.header = _SectionHeader(self)
        self.header.clicked.connect(lambda: self.set_expanded(not self._expanded))
        self.content = QWidget(self)
        self.content.setObjectName("collapsible_content")
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(18, 2, 0, 4)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.header)
        layout.addWidget(self.content)
        self._apply()

    def setTitle(self, title: str) -> None:  # noqa: N802 - matches QGroupBox so the texts table can set it
        """Caption of the header."""
        self._title = title
        self._apply()

    def is_expanded(self) -> bool:
        """Whether the content is shown."""
        return self._expanded

    def set_expanded(self, expanded: bool) -> None:
        """Show or hide the content."""
        if expanded != self._expanded:
            self._expanded = expanded
            self._apply()
            self.toggled.emit(expanded)

    def _apply(self) -> None:
        """Chevron and visibility follow the expanded state."""
        self.header.title.setText(f"{_EXPANDED if self._expanded else _COLLAPSED}  {self._title}")
        self.content.setVisible(self._expanded)
