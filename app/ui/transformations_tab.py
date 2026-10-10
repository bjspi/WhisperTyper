"""Editor for the prompt list: reorderable entries plus the caption/options/prompt fields.

The list holds the user's rephrasing templates and the instruction (LivePrompt) entry. The
instruction entry cannot be removed; it can be moved and switched off, and shows its trigger
settings instead of the template options.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QDropEvent, QPalette
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QWidget,
)

from app.core.i18n import TranslationManager
from app.core.prompts import (
    DEFAULT_INSTRUCTION_CAPTIONS,
    DEFAULT_LIVEPROMPT_SYSTEM_PROMPTS,
    INSTRUCTION,
    MAX_TRANSFORMATIONS,
    _default_prompt_for,
    _is_known_default_prompt,
    load_transformations,
    transformation_entry,
)

_ENTRY_ID = Qt.ItemDataRole.UserRole
#: Pages of the options stack.
class InstructionControls(NamedTuple):
    """Widgets shown only for the instruction entry."""

    #: Switches the whole entry on or off (trigger word, palette, rephrase window).
    active: QCheckBox
    #: LivePrompting by trigger word; the block below the prompt text holds its settings.
    options: QWidget
    trigger_enabled: QCheckBox
    trigger_words: QLineEdit
    scan_depth: QSpinBox
    strip_trigger: QCheckBox
    selection_context: QCheckBox


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
    """Own the prompt list, editing it through the settings page's widgets.

    Entries are keyed by an internal ID stored on the list items, so drag-and-drop
    reordering can never redirect an edit into a different entry.
    """

    #: Emitted after every add, remove, reorder or edit.
    changed = pyqtSignal()

    def __init__(self, entries: Any, translator: TranslationManager, *, splitter: QSplitter,
                 list_placeholder: QWidget, caption_edit: QLineEdit, text_label: QLabel, text_edit: QTextEdit,
                 show_during_recording: QCheckBox, auto_apply: QCheckBox,
                 instruction: InstructionControls, add_button: QPushButton, remove_button: QPushButton,
                 selection_context_supported: bool = True) -> None:
        """Replace the placeholder with a reorderable list and load ``entries``."""
        super().__init__(splitter)
        self.translator = translator
        self.max_entries = MAX_TRANSFORMATIONS
        self._caption_edit = caption_edit
        self._text_label = text_label
        self._text_edit = text_edit
        self._show_during_recording = show_during_recording
        self._auto_apply = auto_apply
        self._instruction = instruction
        self._add_button = add_button
        self._remove_button = remove_button
        self._selection_context_supported = selection_context_supported
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._current_id: Optional[str] = None
        self._loading = False

        self.list = _ReorderableList()
        splitter.replaceWidget(0, self.list)
        list_placeholder.deleteLater()
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        # Selected-text context needs permissions macOS does not grant reliably.
        instruction.selection_context.setVisible(selection_context_supported)

        for entry in load_transformations(entries):
            self._entries[uuid.uuid4().hex] = entry
        for entry_id in self._entries:
            self._append_item(entry_id)
        self.list.currentRowChanged.connect(self._on_selection_changed)
        self.list.reordered.connect(self._on_reordered)
        auto_apply.toggled.connect(self._on_auto_apply_toggled)
        instruction.active.toggled.connect(self._on_instruction_toggled)
        instruction.trigger_enabled.toggled.connect(self._on_instruction_toggled)
        add_button.clicked.connect(self._add)
        remove_button.clicked.connect(self._remove)
        self._load_editor(None)
        if self.list.count():
            self.list.setCurrentRow(0)

    def entries(self) -> List[Dict[str, Any]]:
        """Commit the visible edits and return the entries in display order."""
        self._commit_editor()
        return [transformation_entry(self._entries[entry_id]) for entry_id in self._ordered_ids()]

    def retranslate(self) -> None:
        """Refresh texts that depend on the selected entry and the list placeholders."""
        current = self._entries.get(self._current_id) if self._current_id else None
        self._update_text_label(current)
        for entry_id in self._entries:
            self._refresh_item(entry_id)

    def apply_language_defaults(self, lang_code: str) -> None:
        """Switch the instruction's caption and prompt to ``lang_code`` while they are unedited defaults."""
        self._commit_editor()
        for entry in self._entries.values():
            if entry["kind"] != INSTRUCTION:
                continue
            for field, defaults in (("caption", DEFAULT_INSTRUCTION_CAPTIONS), ("text", DEFAULT_LIVEPROMPT_SYSTEM_PROMPTS)):
                if _is_known_default_prompt(defaults, entry[field]):
                    entry[field] = _default_prompt_for(defaults, lang_code)
        self._load_editor(self._current_id)
        self.retranslate()
        self.changed.emit()

    def _ordered_ids(self) -> List[str]:
        """Entry IDs in the list's visual order."""
        ids = []
        for row in range(self.list.count()):
            item = self.list.item(row)
            if item is not None:
                ids.append(str(item.data(_ENTRY_ID)))
        return ids

    def _own_template_count(self) -> int:
        """Number of the user's templates; the instruction entry does not count against the limit."""
        return sum(1 for entry in self._entries.values() if entry["kind"] != INSTRUCTION)

    def _label(self, entry: Dict[str, Any]) -> str:
        """List label; empty captions show the translated placeholder."""
        return str(entry.get("caption", "")).strip() or self.translator.tr("caption_placeholder")

    def _append_item(self, entry_id: str) -> None:
        """Add the list row representing one entry."""
        item = QListWidgetItem()
        item.setData(_ENTRY_ID, entry_id)
        self.list.addItem(item)
        self._refresh_item(entry_id)

    def _refresh_item(self, entry_id: str) -> None:
        """Update a row's label; a switched-off instruction is shown greyed out."""
        item = self._item_for(entry_id)
        if item is None:
            return
        entry = self._entries[entry_id]
        item.setText(self._label(entry))
        inactive = entry["kind"] == INSTRUCTION and not entry["enabled"]
        group = QPalette.ColorGroup.Disabled if inactive else QPalette.ColorGroup.Normal
        item.setForeground(QBrush(self.list.palette().color(group, QPalette.ColorRole.Text)))

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
                     show_during_recording=self._show_during_recording.isChecked())
        if entry["kind"] == INSTRUCTION:
            controls = self._instruction
            entry.update(enabled=controls.active.isChecked(), trigger_enabled=controls.trigger_enabled.isChecked(),
                         trigger_words=controls.trigger_words.text(),
                         scan_depth=controls.scan_depth.value(), strip_trigger=controls.strip_trigger.isChecked(),
                         use_selection_context=self._selection_context_supported
                         and controls.selection_context.isChecked())
        else:
            entry["auto_apply"] = self._auto_apply.isChecked()
        self._refresh_item(self._current_id)

    def _update_text_label(self, entry: Optional[Dict[str, Any]]) -> None:
        """The instruction's text is its system prompt; templates keep the generic label."""
        is_instruction = bool(entry and entry["kind"] == INSTRUCTION)
        self._text_label.setText(self.translator.tr("liveprompt_system_prompt_label" if is_instruction
                                                    else "text_label"))

    def _load_editor(self, entry_id: Optional[str]) -> None:
        """Show one entry in the editor fields, or clear and disable them."""
        self._loading = True
        entry = self._entries.get(entry_id) if entry_id else None
        is_instruction = bool(entry and entry["kind"] == INSTRUCTION)
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
        if entry is not None and is_instruction:
            controls = self._instruction
            controls.active.setChecked(entry["enabled"])
            controls.trigger_enabled.setChecked(entry["trigger_enabled"])
            controls.trigger_words.setText(entry["trigger_words"])
            controls.scan_depth.setValue(entry["scan_depth"])
            controls.strip_trigger.setChecked(entry["strip_trigger"])
            controls.selection_context.setChecked(entry["use_selection_context"])
        # Hidden widgets take no room, so each kind shows a compact form with the text filling the rest.
        self._auto_apply.setVisible(not is_instruction)
        self._instruction.active.setVisible(is_instruction)
        self._instruction.options.setVisible(is_instruction)
        self._update_text_label(entry)
        self._current_id = entry_id if entry else None
        self._loading = False
        self._update_instruction_states()
        self._update_buttons()

    def _update_instruction_states(self) -> None:
        """Grey out what a switched-off instruction, or switched-off LivePrompting, does not use."""
        entry = self._entries.get(self._current_id) if self._current_id else None
        if entry is None or entry["kind"] != INSTRUCTION:
            return
        controls = self._instruction
        active = controls.active.isChecked()
        entry_widgets: Tuple[QWidget, ...] = (self._caption_edit, self._text_edit, self._show_during_recording,
                                              controls.trigger_enabled, controls.selection_context)
        for widget in entry_widgets:
            widget.setEnabled(active)
        triggered = active and controls.trigger_enabled.isChecked()
        trigger_widgets: Tuple[QWidget, ...] = (controls.trigger_words, controls.scan_depth, controls.strip_trigger)
        for widget in trigger_widgets:
            widget.setEnabled(triggered)

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

    def _on_instruction_toggled(self, _checked: bool) -> None:
        """Switching the instruction or its trigger takes effect (and greys its row) right away."""
        if self._loading:
            return
        self._update_instruction_states()
        self._commit_editor()
        self.changed.emit()

    def _update_buttons(self) -> None:
        """Respect the template limit; the instruction entry cannot be removed."""
        current = self._entries.get(self._current_id) if self._current_id else None
        self._add_button.setEnabled(self._own_template_count() < self.max_entries)
        self._remove_button.setEnabled(current is not None and current["kind"] != INSTRUCTION)

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
        if self._own_template_count() >= self.max_entries:
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
        if item is None or self._entries[str(item.data(_ENTRY_ID))]["kind"] == INSTRUCTION:
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
