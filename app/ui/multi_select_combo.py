"""A dropdown whose entries are ticked on and off, looking like every other dropdown of the app."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

from PyQt6.QtCore import QEvent, QModelIndex, QObject, QPointF, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QMouseEvent, QPalette, QPixmap, QStandardItemModel
from PyQt6.QtWidgets import QComboBox, QStyle, QStyleOptionComboBox, QStylePainter, QWidget

from app.ui.line_icons import line_icon

#: Entries shown before the list scrolls; the full language catalog would cover the screen.
_MAX_VISIBLE_ENTRIES = 12
#: Size of the tick in front of a chosen entry; unticked entries keep an empty slot so captions align.
_TICK_SIZE = 16
#: Item data role holding whether an entry is ticked.
_CHECKED_ROLE = Qt.ItemDataRole.UserRole + 1
#: Gap between the pieces of a summary made of flags and captions.
_SUMMARY_SPACING = 6

#: One piece of the closed box's summary: a caption or a small picture such as a flag.
SummaryPart = Union[str, QPixmap]


class MultiSelectComboBox(QComboBox):
    """Dropdown with several chosen entries: a click ticks or unticks an entry and the list stays open.

    The closed box shows a caller-provided summary of the choice instead of a current entry. The
    popup takes the stylesheet's dropdown look (``theme.round_combo_popups`` may replace the view,
    which is why the click filter is attached whenever the list opens).
    """

    #: An entry was ticked or unticked by the user.
    selection_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """An empty dropdown; fill it with ``add_choice``."""
        super().__init__(parent)
        self.setMaxVisibleItems(_MAX_VISIBLE_ENTRIES)
        self._summary: List[SummaryPart] = []
        self._tick = QIcon()
        blank = QPixmap(_TICK_SIZE, _TICK_SIZE)
        blank.fill(Qt.GlobalColor.transparent)
        self._no_tick = QIcon(blank)

    # --- Entries --------------------------------------------------------------------------
    def add_choice(self, caption: str, value: str) -> None:
        """Append an unticked entry."""
        self.addItem(self._no_tick, caption, value)
        self.setItemData(self.count() - 1, False, _CHECKED_ROLE)

    def checked_values(self) -> List[str]:
        """The values of the ticked entries, in list order."""
        return [self.itemData(index) for index in range(self.count()) if self.itemData(index, _CHECKED_ROLE)]

    def set_checked_values(self, values: List[str]) -> None:
        """Tick exactly the entries whose value is in ``values``."""
        chosen = set(values)
        for index in range(self.count()):
            self._set_checked(index, self.itemData(index) in chosen)

    def set_enabled_values(self, enabled: Dict[str, bool]) -> None:
        """Let the user change only the entries enabled here; values left out keep their state."""
        for index in range(self.count()):
            value = self.itemData(index)
            if value in enabled:
                self._set_item_enabled(index, enabled[value])

    def set_caption(self, value: str, caption: str) -> None:
        """Rename the entry holding ``value`` (after a language switch of the UI)."""
        index = self.findData(value)
        if index >= 0:
            self.setItemText(index, caption)

    def set_summary(self, parts: Sequence[SummaryPart]) -> None:
        """What the closed box shows, e.g. the chosen entries in short: captions and pictures in a row."""
        self._summary = list(parts)
        self.update()

    # --- Behaviour ------------------------------------------------------------------------
    def showPopup(self) -> None:  # noqa: N802 - Qt API
        """Open the list with ticks in the theme's colours and clicks that toggle instead of closing."""
        view = self.view()
        if view is not None:
            viewport = view.viewport()
            if viewport is not None:
                viewport.installEventFilter(self)  # installing twice keeps a single filter
            view.ensurePolished()
            self._tint_ticks(view.palette())
        super().showPopup()

    def eventFilter(self, watched: Optional[QObject], event: Optional[QEvent]) -> bool:  # noqa: N802 - Qt API
        """Toggle the clicked entry; swallowing the release keeps the list open."""
        if event is None or event.type() != QEvent.Type.MouseButtonRelease or not isinstance(event, QMouseEvent):
            return super().eventFilter(watched, event)
        view = self.view()
        index = view.indexAt(event.position().toPoint()) if view is not None else QModelIndex()
        if index.isValid() and index.flags() & Qt.ItemFlag.ItemIsEnabled:
            self._set_checked(index.row(), not self.itemData(index.row(), _CHECKED_ROLE))
            self.selection_changed.emit()
        return True

    def paintEvent(self, _event: object) -> None:  # noqa: N802 - Qt API
        """Draw the box with the summary in place of a current entry."""
        painter = QStylePainter(self)
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        option.currentText = ""
        option.currentIcon = QIcon()
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, option)
        style = self.style()
        if style is None:
            return
        field = style.subControlRect(QStyle.ComplexControl.CC_ComboBox, option,
                                     QStyle.SubControl.SC_ComboBoxEditField, self)
        self._paint_summary(painter, field)

    def _paint_summary(self, painter: QStylePainter, field: QRect) -> None:
        """Draw the summary's captions and pictures left to right, vertically centred in ``field``."""
        painter.setClipRect(field)
        painter.setPen(self.palette().color(QPalette.ColorRole.ButtonText))
        x = field.left()
        for part in self._summary:
            if isinstance(part, QPixmap):
                size = part.deviceIndependentSize()
                painter.drawPixmap(QPointF(x, field.top() + (field.height() - size.height()) / 2), part)
                x += round(size.width())
            else:
                width = self.fontMetrics().horizontalAdvance(part)
                painter.drawText(QRect(x, field.top(), width, field.height()), Qt.AlignmentFlag.AlignVCenter, part)
                x += width
            x += _SUMMARY_SPACING

    # --- Internals ------------------------------------------------------------------------
    def _set_checked(self, index: int, checked: bool) -> None:
        """Store the tick and show it as the entry's icon."""
        self.setItemData(index, checked, _CHECKED_ROLE)
        self.setItemIcon(index, self._tick if checked else self._no_tick)

    def _set_item_enabled(self, index: int, enabled: bool) -> None:
        """Grey out an entry the user may not change right now."""
        model = self.model()  # a combo box's own model is a QStandardItemModel
        item = model.item(index) if isinstance(model, QStandardItemModel) else None
        if item is not None:
            item.setEnabled(enabled)

    def _tint_ticks(self, colors: QPalette) -> None:
        """Draw the ticks in the list's accent colour, and in its contrast colour on the hovered row."""
        accent = colors.color(QPalette.ColorRole.Highlight).name()
        on_accent = colors.color(QPalette.ColorRole.HighlightedText).name()
        self._tick = line_icon("check", accent, on_accent)
        for index in range(self.count()):
            self._set_checked(index, bool(self.itemData(index, _CHECKED_ROLE)))
