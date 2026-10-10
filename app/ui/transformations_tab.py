"""Editor for the transformation templates: reorderable list plus caption/prompt fields."""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QDropEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTextEdit,
    QWidget,
)

from app.core.i18n import TranslationManager
from app.core.prompts import MAX_TRANSFORMATIONS, load_transformations, transformation_entry

_ENTRY_ID = Qt.ItemDataRole.UserRole


class _ReorderableList(QListWidget):
    """Single-selection list whose drag-and-drop reordering is reported through ``reordered``."""

    reordered = pyqtSignal()

    def __init__(self) -> None:
        """Enable internal move drag-and-drop."""
        super().__init__()
        self.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)

    def dropEvent(self, event: Optional[QDropEvent]) -> None:  # noqa: N802 - Qt API
        """Report the new visual order once Qt has moved the item."""
        super().dropEvent(event)
        self.reordered.emit()


class TransformationsEditor(QObject):
    """Own the template list, editing it through the settings page's widgets.

    Entries are keyed by an internal ID stored on the list items, so drag-and-drop
    reordering can never redirect an edit into a different template.
    """

    #: Emitted after every add, remove, reorder or edit.
    changed = pyqtSignal()

    def __init__(self, entries: Any, translator: TranslationManager, *, splitter: QSplitter,
                 list_placeholder: QWidget, caption_edit: QLineEdit, text_edit: QTextEdit,
                 show_during_recording: QCheckBox, auto_apply: QCheckBox, add_button: QPushButton,
                 remove_button: QPushButton) -> None:
        """Replace the placeholder with a reorderable list and load ``entries``."""
        super().__init__(splitter)
        self.translator = translator
        self.max_entries = MAX_TRANSFORMATIONS
        self._caption_edit = caption_edit
        self._text_edit = text_edit
        self._show_during_recording = show_during_recording
        self._auto_apply = auto_apply
        self._add_button = add_button
        self._remove_button = remove_button
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._current_id: Optional[str] = None
        self._loading = False

        self.list = _ReorderableList()
        splitter.replaceWidget(0, self.list)
        list_placeholder.deleteLater()
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        for entry in load_transformations(entries):
            self._entries[uuid.uuid4().hex] = entry
        for entry_id in self._entries:
            self._append_item(entry_id)
        self.list.currentRowChanged.connect(self._on_selection_changed)
        self.list.reordered.connect(self._on_reordered)
        auto_apply.toggled.connect(self._on_auto_apply_toggled)
        add_button.clicked.connect(self._add)
        remove_button.clicked.connect(self._remove)
        self._load_editor(None)
        if self.list.count():
            self.list.setCurrentRow(0)

    def entries(self) -> List[Dict[str, Any]]:
        """Commit the visible edits and return the templates in display order."""
        self._commit_editor()
        return [transformation_entry(self._entries[entry_id]) for entry_id in self._ordered_ids()]

    def _ordered_ids(self) -> List[str]:
        """Entry IDs in the list's visual order."""
        ids = []
        for row in range(self.list.count()):
            item = self.list.item(row)
            if item is not None:
                ids.append(str(item.data(_ENTRY_ID)))
        return ids

    def _label(self, entry: Dict[str, Any]) -> str:
        """List label; empty captions show the translated placeholder."""
        return str(entry.get("caption", "")).strip() or self.translator.tr("caption_placeholder")

    def _append_item(self, entry_id: str) -> None:
        """Add the list row representing one entry."""
        item = QListWidgetItem(self._label(self._entries[entry_id]))
        item.setData(_ENTRY_ID, entry_id)
        self.list.addItem(item)

    def _item_for(self, entry_id: Optional[str]) -> Optional[QListWidgetItem]:
        """Return the list row for an entry, wherever it was dragged to."""
        for row in range(self.list.count()):
            item = self.list.item(row)
            if item is not None and item.data(_ENTRY_ID) == entry_id:
                return item
        return None

    def _commit_editor(self) -> None:
        """Write the editor fields back into the entry they were loaded from."""
        if self._loading or self._current_id not in self._entries:
            return
        assert self._current_id is not None
        entry = self._entries[self._current_id]
        entry.update(caption=self._caption_edit.text(), text=self._text_edit.toPlainText(),
                     show_during_recording=self._show_during_recording.isChecked(),
                     auto_apply=self._auto_apply.isChecked())
        item = self._item_for(self._current_id)
        if item is not None:
            item.setText(self._label(entry))

    def _load_editor(self, entry_id: Optional[str]) -> None:
        """Show one entry in the editor fields, or clear and disable them."""
        self._loading = True
        entry = self._entries.get(entry_id) if entry_id else None
        auto_apply = bool(entry and entry.get("auto_apply") is True)
        for widget in (self._caption_edit, self._text_edit, self._auto_apply):
            widget.setEnabled(entry is not None)
        # The automatic prompt is always offered in the palette, so it can be deselected there.
        self._show_during_recording.setEnabled(entry is not None and not auto_apply)
        # Deliberately no focus change: this runs on every selection change, and moving focus
        # into the editor would break arrow-key navigation through the list.
        self._caption_edit.setText(str(entry.get("caption", "")) if entry else "")
        self._text_edit.setPlainText(str(entry.get("text", "")) if entry else "")
        self._show_during_recording.setChecked(bool(entry and entry.get("show_during_recording") is True))
        self._auto_apply.setChecked(auto_apply)
        self._current_id = entry_id if entry else None
        self._loading = False
        self._update_buttons()

    def _on_auto_apply_toggled(self, checked: bool) -> None:
        """Only one template applies automatically, and it is always shown during recording."""
        if self._loading or self._current_id not in self._entries:
            return
        if checked:
            self._show_during_recording.setChecked(True)
            for entry_id, entry in self._entries.items():
                if entry_id != self._current_id:
                    entry["auto_apply"] = False
        self._show_during_recording.setEnabled(not checked)
        self._commit_editor()
        self.changed.emit()

    def _update_buttons(self) -> None:
        """Respect the template limit and require a selection for removal."""
        self._add_button.setEnabled(len(self._entries) < self.max_entries)
        self._remove_button.setEnabled(bool(self._entries) and self.list.currentRow() >= 0)

    def _on_selection_changed(self, row: int) -> None:
        """Save the previous entry, then load the newly selected one."""
        if self._loading:
            return
        self._commit_editor()
        item = self.list.item(row) if row >= 0 else None
        self._load_editor(str(item.data(_ENTRY_ID)) if item is not None else None)
        self.changed.emit()

    def _on_reordered(self) -> None:
        """Keep editing the same entry after a drag-and-drop."""
        self._commit_editor()
        item = self.list.currentItem()
        self._load_editor(str(item.data(_ENTRY_ID)) if item is not None else None)
        self.changed.emit()

    def _add(self) -> None:
        """Append a blank template and focus its caption for immediate typing."""
        if len(self._entries) >= self.max_entries:
            return
        self._commit_editor()
        entry_id = uuid.uuid4().hex
        self._entries[entry_id] = transformation_entry({})
        self._append_item(entry_id)
        self.list.setCurrentRow(self.list.count() - 1)
        self._caption_edit.setFocus()
        self.changed.emit()

    def _remove(self) -> None:
        """Delete the selected template and select its neighbour."""
        row = self.list.currentRow()
        item = self.list.item(row) if row >= 0 else None
        if item is None:
            return
        self._loading = True
        del self._entries[str(item.data(_ENTRY_ID))]
        self._current_id = None
        self.list.takeItem(row)
        self._loading = False
        next_row = min(row, self.list.count() - 1)
        if next_row >= 0:
            self.list.setCurrentRow(next_row)
            current = self.list.item(next_row)
            self._load_editor(str(current.data(_ENTRY_ID)) if current is not None else None)
        else:
            self._load_editor(None)
        self.changed.emit()
