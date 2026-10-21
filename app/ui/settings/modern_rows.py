"""Modern design: settings cards as rows - title and description left, the control right.

The rows reuse the window's existing widgets (loaded from the .ui file or built by the pages), so
bindings, signals and tests keep working; only their layout changes. A row hides with its control
and greys its title with it, so the pages' own show/hide/enable logic carries over unchanged.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QBoxLayout,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QSlider,
    QVBoxLayout,
    QWidget,
)

#: Widths of the controls on the right, so they line up from row to row and card to card.
COMBO_WIDTH = 260
FIELD_WIDTH = 300
NUMBER_WIDTH = 110
SLIDER_WIDTH = 210
CHECKBOX_SIZE = (34, 22)  # the switch drawn by the theme; the checkbox's own text is not shown
ROW_SPACING = 16
CARD_MARGINS = (16, 4, 16, 4)
#: Tooltips up to this length (plain text) are shown as the row's description.
DESCRIPTION_MAX_CHARS = 150

Control = Union[QWidget, int]  # an int adds that much stretch between the controls


class SettingRow(QWidget):
    """Title (and a one-line description) on the left, controls on the right.

    ``follow`` is the control whose visibility and enabled state the row mirrors.
    """

    def __init__(self, title: Optional[QLabel], controls: Sequence[Control], *, follow: Optional[QWidget] = None,
                 checkbox: Optional[QCheckBox] = None, stacked: bool = False, plain_title: bool = False) -> None:
        """``plain_title`` keeps the title label's own text and style (a status line, not a caption)."""
        super().__init__()
        self.setObjectName("settingRow")
        self.checkbox = checkbox
        self._plain_title = plain_title
        self.title = title if title is not None else (QLabel() if checkbox is not None else None)
        self.description = QLabel()
        self.description.setObjectName("rowDescription")
        self.description.setWordWrap(True)
        self.description.hide()
        self.follow = follow or checkbox or next((c for c in controls if isinstance(c, QWidget)), None)

        texts = QVBoxLayout()
        texts.setSpacing(2)
        if self.title is not None:
            if not plain_title:
                self.title.setProperty("rowTitle", True)
            self.title.setWordWrap(True)
            texts.addWidget(self.title)
        texts.addWidget(self.description)
        line = QHBoxLayout()
        line.setSpacing(8)
        for control in controls:
            if isinstance(control, int):
                line.addStretch(control)
            else:
                _fit(control)
                line.addWidget(control)
        layout: QBoxLayout
        if stacked:
            # Wide content (a status text, a full-width field) goes below the title.
            layout = QVBoxLayout(self)
            layout.setSpacing(8)
            layout.addLayout(texts)
            layout.addLayout(line)
        else:
            layout = QHBoxLayout(self)
            layout.setSpacing(ROW_SPACING)
            layout.addLayout(texts, 1)
            layout.addLayout(line)
        layout.setContentsMargins(0, 10, 0, 10)
        if checkbox is not None:
            checkbox.setFixedSize(*CHECKBOX_SIZE)
            checkbox.setCursor(Qt.CursorShape.PointingHandCursor)
        if self.follow is not None:
            # Only ever hide here: showing a row before a card adopts it would open it as a window.
            if self.follow.isHidden():
                self.hide()
            self._sync_enabled()
            self.follow.installEventFilter(self)

    def eventFilter(self, watched: Optional[QObject], event: Optional[QEvent]) -> bool:  # noqa: N802 - Qt API
        """Mirror the followed control's explicit show/hide and its enabled state."""
        if watched is self.follow and event is not None:
            kind = event.type()
            if kind == QEvent.Type.ShowToParent:
                self.show()
            elif kind == QEvent.Type.HideToParent:
                self.hide()
            elif kind == QEvent.Type.EnabledChange:
                self._sync_enabled()
        return False

    def _sync_enabled(self) -> None:
        """Grey the texts while the control is disabled."""
        enabled = self.follow is None or self.follow.isEnabled()
        if self.title is not None:
            self.title.setEnabled(enabled)
        self.description.setEnabled(enabled)

    def sync_texts(self) -> None:
        """Take the title from the checkbox or the caption (without its colon), the description from the tooltip."""
        if self.title is None or self._plain_title:
            return
        if self.checkbox is not None:
            self.title.setText(self.checkbox.text())
        else:
            self.title.setText(self.title.text().rstrip().rstrip(":").rstrip())
        source = self.checkbox if self.checkbox is not None else self.title
        tip = source.toolTip().strip()
        plain = bool(tip) and "<" not in tip and "\n" not in tip and len(tip) <= DESCRIPTION_MAX_CHARS
        self.description.setText(tip if plain else "")
        self.description.setVisible(plain)
        if self.checkbox is not None:
            self.setToolTip(self.checkbox.toolTip() if not plain else "")

    def mouseReleaseEvent(self, event: Optional[QMouseEvent]) -> None:  # noqa: N802 - Qt API
        """A click anywhere on a switch row toggles it, like a click on its label would."""
        if (event is not None and event.button() == Qt.MouseButton.LeftButton and self.checkbox is not None
                and self.checkbox.isEnabled()):
            self.checkbox.click()
            return
        super().mouseReleaseEvent(event)


class CardBody(QWidget):
    """The rows of one card, separated by hairlines that skip hidden rows."""

    def __init__(self, parent: QWidget, margins: Tuple[int, int, int, int] = CARD_MARGINS) -> None:
        """An empty body inside ``parent``, padded by ``margins``."""
        super().__init__(parent)
        self.setObjectName("cardBody")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(*margins)
        self._layout.setSpacing(0)
        self._entries: List[Tuple[QFrame, QWidget]] = []

    def add(self, widget: QWidget) -> None:
        """Append a row (or any widget, such as a collapsible section) below a hairline."""
        divider = QFrame(self)
        divider.setObjectName("rowDivider")
        divider.setFixedHeight(1)
        self._layout.addWidget(divider)
        self._layout.addWidget(widget)
        self._entries.append((divider, widget))
        widget.installEventFilter(self)
        self._update_dividers()

    def set_stretch(self, widget: QWidget, stretch: int) -> None:
        """Let ``widget`` take the free height (such as a prompt text field)."""
        self._layout.setStretchFactor(widget, stretch)

    def eventFilter(self, watched: Optional[QObject], event: Optional[QEvent]) -> bool:  # noqa: N802 - Qt API
        """Re-place the hairlines whenever a row is shown or hidden."""
        if event is not None and event.type() in (QEvent.Type.ShowToParent, QEvent.Type.HideToParent):
            self._update_dividers()
        return False

    def _update_dividers(self) -> None:
        """A hairline only between two visible rows."""
        seen_visible = False
        for divider, widget in self._entries:
            visible = not widget.isHidden()
            divider.setVisible(visible and seen_visible)
            seen_visible = seen_visible or visible


def clear_layout(layout: QLayout) -> None:
    """Empty ``layout`` and its nested layouts; the widgets stay alive (their parent is unchanged)."""
    while layout.count():
        item = layout.takeAt(0)
        if item is None:
            continue
        nested = item.layout()
        if nested is not None:
            clear_layout(nested)
            nested.setParent(None)


def rebuild_card(owner: QWidget, rows: Sequence[QWidget],
                 margins: Tuple[int, int, int, int] = CARD_MARGINS) -> CardBody:
    """Replace the content of ``owner``'s layout with ``rows`` in a card body."""
    layout = owner.layout()
    assert layout is not None
    clear_layout(layout)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    body = CardBody(owner, margins)
    for row in rows:
        body.add(row)
    layout.addWidget(body)
    return body


def _fit(control: QWidget) -> None:
    """Give the controls on the right their common widths (a narrower maximum set beforehand wins)."""
    for kind, width in ((QComboBox, COMBO_WIDTH), (QAbstractSpinBox, NUMBER_WIDTH), (QLineEdit, FIELD_WIDTH),
                        (QSlider, SLIDER_WIDTH)):
        if isinstance(control, kind):
            control.setFixedWidth(min(width, control.maximumWidth()))
            if isinstance(control, QAbstractSpinBox):
                control.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return
