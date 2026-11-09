"""The search field above the sidebar pages and the list of matching settings.

Typing replaces the page list with the matches; choosing one opens its page, scrolls to the
setting and lights it up for a moment (``ModernChrome`` does the opening).
"""
from __future__ import annotations

from typing import Callable, List, Optional

from PyQt6.QtCore import QEvent, QModelIndex, QObject, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QKeyEvent
from PyQt6.QtWidgets import (
    QFrame,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QWidget,
)

from app.core.settings_search import SettingEntry, find, place
from app.ui.line_icons import line_icon

_ENTRY = Qt.ItemDataRole.UserRole
#: Horizontal padding and border of a sidebar item (``QListWidget#nav::item`` in the theme).
_NAV_ITEM_INSET = 2 * 10 + 2 * 1


class _PaddedWrapDelegate(QStyledItemDelegate):
    """Measures wrapped text at the width it is drawn in.

    Qt wraps the text for the size hint at the full item width but draws it inside the
    stylesheet padding, so a title wrapping onto one more line lost its page to an ellipsis.
    """

    def sizeHint(self, option: QStyleOptionViewItem,  # noqa: N802 - Qt API
                 index: QModelIndex) -> QSize:
        """The size for the item's text wrapped inside the padding."""
        inner = QStyleOptionViewItem(option)
        inner.rect.setWidth(option.rect.width() - _NAV_ITEM_INSET)
        return super().sizeHint(inner, index)


class SettingsSearch(QObject):
    """The search field and its result list; the sidebar places both, the window opens a choice."""

    #: The SettingEntry to open.
    chosen = pyqtSignal(object)
    #: Results replace the page list while text is typed.
    active_changed = pyqtSignal(bool)

    def __init__(self, entries: Callable[[], List[SettingEntry]], tr: Callable[[str], str],
                 parent: QWidget) -> None:
        """``entries`` is asked at each keystroke, so settings shown meanwhile are found too."""
        super().__init__(parent)
        self._entries = entries
        self._tr = tr
        self.active = False
        self.field = QLineEdit(parent)
        self.field.setObjectName("settingsSearch")
        self.field.setClearButtonEnabled(True)
        self._magnifier = self.field.addAction(QIcon(), QLineEdit.ActionPosition.LeadingPosition)
        self.field.textChanged.connect(self._search)
        self.field.installEventFilter(self)
        self.results = QListWidget(parent)
        self.results.setObjectName("nav")
        self.results.setFrameShape(QFrame.Shape.NoFrame)
        self.results.setWordWrap(True)  # long titles wrap instead of scrolling sideways
        self.results.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Like the page list: no scroll bar (the wheel scrolls), so the width the delegate measures stays put.
        self.results.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.results.setItemDelegate(_PaddedWrapDelegate(self.results))
        self.results.hide()
        self.results.itemClicked.connect(self._open)
        self.results.itemActivated.connect(self._open)

    def set_icon_color(self, color: str) -> None:
        """Tint the magnifier for the active theme."""
        if self._magnifier is not None:
            self._magnifier.setIcon(line_icon("search", color))

    def retranslate(self) -> None:
        """Placeholder and accessible names in the UI language; shown results follow it too."""
        self.field.setPlaceholderText(self._tr("settings_search_placeholder"))
        self.field.setAccessibleName(self._tr("settings_search_placeholder"))
        self.results.setAccessibleName(self._tr("settings_search_results"))
        if self.active:
            self._search(self.field.text())

    def focus(self) -> None:
        """Put the cursor into the field, its text selected (Ctrl+F)."""
        self.field.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self.field.selectAll()

    def eventFilter(self, watched: Optional[QObject], event: Optional[QEvent]) -> bool:  # noqa: N802 - Qt API
        """In the field, Enter opens the selected match, Esc clears typed text and Down moves into the results.

        Enter is consumed, so it never reaches a default button.
        """
        if watched is not self.field or not isinstance(event, QKeyEvent) or event.type() != QEvent.Type.KeyPress:
            return False
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._open_current()
            return True
        if key == Qt.Key.Key_Escape and self.field.text():
            self.field.clear()
            return True
        if key == Qt.Key.Key_Down and self.active:
            self.results.setFocus(Qt.FocusReason.OtherFocusReason)
            return True
        return False

    def _search(self, text: str) -> None:
        """List the matches, the first one selected, or say that none match."""
        active = bool(text.strip())
        if active != self.active:
            self.active = active
            self.results.setVisible(active)
            self.active_changed.emit(active)
        self.results.clear()
        if not active:
            return
        entries = self._entries()
        for entry in find(text, entries):
            self._add_result(entry, place(entry, entries))
        if self.results.count():
            self.results.setCurrentRow(0)
            return
        item = QListWidgetItem(self._tr("settings_search_no_match"))
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        self.results.addItem(item)

    def _add_result(self, entry: SettingEntry, where: str) -> None:
        """The setting's title with where it lives on a second line; a page alone for a page.

        Plain two-line items: widgets per item overlapped their texts.
        """
        item = QListWidgetItem(entry.page if entry.target is None else f"{entry.title}\n{where}")
        item.setData(_ENTRY, entry)
        item.setToolTip(entry.description)
        self.results.addItem(item)

    def _open_current(self) -> None:
        """Open the selected match, if there is one."""
        item = self.results.currentItem()
        if item is not None:
            self._open(item)

    def _open(self, item: Optional[QListWidgetItem]) -> None:
        """Announce the entry behind ``item``."""
        entry = item.data(_ENTRY) if item is not None else None
        if isinstance(entry, SettingEntry):
            self.chosen.emit(entry)
