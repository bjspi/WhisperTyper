"""Wrapping layout: items run left to right and continue on the next line when the width is used up."""
from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import QMargins, QPoint, QRect, QSize, Qt
from PyQt6.QtWidgets import QLayout, QLayoutItem, QWidget


class FlowLayout(QLayout):
    """Qt's flow layout: a height-for-width layout that wraps its items like words in a paragraph."""

    def __init__(self, parent: Optional[QWidget] = None, spacing: int = 6) -> None:
        """Create an empty flow with uniform horizontal and vertical ``spacing``."""
        super().__init__(parent)
        self._items: List[QLayoutItem] = []
        self._spacing = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item: Optional[QLayoutItem]) -> None:  # noqa: N802 - Qt API
        """Append an item; widgets added with ``addWidget`` arrive here."""
        if item is not None:
            self._items.append(item)

    def count(self) -> int:
        """Number of items in the flow."""
        return len(self._items)

    def itemAt(self, index: int) -> Optional[QLayoutItem]:  # noqa: N802 - Qt API
        """Item at ``index``, or None past the end."""
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> Optional[QLayoutItem]:  # noqa: N802 - Qt API
        """Remove and return the item at ``index``."""
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Qt.Orientation:  # noqa: N802 - Qt API
        """The flow never claims extra space; its height follows from the width."""
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt API
        """The height depends on how many lines the width allows."""
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt API
        """Height needed to lay out all items within ``width``."""
        return self._arrange(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802 - Qt API
        """Place the items inside ``rect``."""
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        """Preferred size: everything on one line."""
        margins = self.contentsMargins()
        width = sum(item.sizeHint().width() for item in self._items) + self._spacing * max(0, len(self._items) - 1)
        height = max((item.sizeHint().height() for item in self._items), default=0)
        return QSize(width + margins.left() + margins.right(), height + margins.top() + margins.bottom())

    def minimumSize(self) -> QSize:  # noqa: N802 - Qt API
        """Smallest size: the widest single item."""
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins: QMargins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(), margins.top() + margins.bottom())

    def _arrange(self, rect: QRect, apply: bool) -> int:
        """Lay out (or only measure) the items line by line; returns the total height."""
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        x, y, line_height = area.x(), area.y(), 0
        for item in self._items:
            hint = item.sizeHint()
            if x > area.x() and x + hint.width() > area.right() + 1:
                x, y, line_height = area.x(), y + line_height + self._spacing, 0
            if apply:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + self._spacing
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y() + margins.bottom()
