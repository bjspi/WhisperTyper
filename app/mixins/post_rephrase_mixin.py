"""PostRephraseMixin — post-rephrase transformations editor and trigger."""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

import copykitten
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QListWidgetItem, QWidget

from app.core.constants import activate_app, get_active_app_name
from app.core.env import is_MACOS


class PostRephraseMixin:
    """Post-rephrase transformations editor and trigger."""

    def trigger_post_rephrase_window(self) -> None:
        """Checks for selected text and shows the floating button window if text is present."""
        # Check for API settings before proceeding
        if not self.config.get("rephrasing_api_url") or not self.config.get("rephrasing_api_key") or not self.config.get("rephrasing_model"):
            logging.warning("Post-rephrase hotkey pressed, but API settings are missing.")
            self.show_tray_balloon(self.translator.tr("rephrase_api_settings_missing"), 3000)
            return

        if is_MACOS:
            logging.debug("Trying to get currently active window/application on macOS via osascript")
            try:
                self.macos_active_application = get_active_app_name()
            except Exception as e:
                # osascript can fail (e.g. CalledProcessError) when Automation
                # permission is missing; on_rephrasing_finished handles None.
                logging.warning(f"Could not determine active macOS application: {e}")
                self.macos_active_application = None

        selected_text = self.get_selected_text(
            select_all_first=self.config.get("post_rephrase_auto_select_all", False)
        )
        if not selected_text:
            logging.info("Post-rephrase hotkey pressed, but no text was selected.")
            self.show_tray_balloon(self.translator.tr("no_text_selected_for_rephrase"), 2000)
            return

        # Filter for entries that have a caption
        valid_entries = [entry for entry in self.config.get("post_rephrasing_entries", []) if entry.get("caption", "").strip()]
        if not valid_entries:
            logging.info("Post-rephrase hotkey pressed, but no valid post-processing entries are configured.")
            self.show_tray_balloon(self.translator.tr("no_post_rephrase_entries_configured"), 3000)
            return

        # Emit a signal to create the window in the main GUI thread
        self.show_floating_window_signal.emit(valid_entries, selected_text)

    def on_floating_button_clicked(self, system_prompt: str, selected_text: str, window: QWidget) -> None:
        """
        Callback executed when a button in the floating window is clicked.

        Args:
            system_prompt (str): The prompt associated with the clicked button.
            selected_text (str): The text that was selected when the window was opened.
            window (QWidget): The floating window instance, to be closed.
        """
        window.close()
        logging.info("Floating button clicked. Rephrasing selected text with custom prompt.")
        # Persistent spinner balloon that stays up for the whole request; ended by the
        # finished/error callbacks (analogous to the transcription/LivePrompt spinner).
        self.show_tray_balloon(self.translator.tr("rephrasing_selection_message"), 0, spinner=True)

        # Context is not used for this specific action.
        worker = self._build_rephrasing_worker(system_prompt, selected_text)
        self._start_rephrasing_worker(
            worker,
            on_finished=self.on_rephrasing_finished,
            on_error=self.on_rephrasing_error,
        )

    def on_rephrasing_finished(self, rephrased_text: str) -> None:
        """
        Callback for when rephrasing from the floating window is successful.

        Args:
            rephrased_text (str): The text returned by the AI.
        """
        # On macOS, pasting can be unreliable if the app loses focus.
        # It's safer to copy to clipboard and notify the user.
        if is_MACOS:
            if self.macos_active_application:
                # Try to reactivate the original app using osascript
                activate_app(self.macos_active_application)
                # Swap the spinner for a brief "done ✓" balloon, then type the text.
                self.show_tray_balloon(self.translator.tr("rephrasing_done_message"), 1600, check=True)
                self.insert_transcribed_text(rephrased_text)
            else: # Fallback in case the app name could not be determined using osascript beforehand
                copykitten.copy(rephrased_text)
                self.show_tray_balloon(self.translator.tr("rephrasing_finished_macos_message"), 3500, check=True)
        else:
            # Swap the spinner for a brief "done ✓" balloon, then type the text.
            self.show_tray_balloon(self.translator.tr("rephrasing_done_message"), 1600, check=True)
            self.insert_transcribed_text(rephrased_text)
            logging.info("Successfully inserted rephrased text.")

    def on_rephrasing_error(self, error_message: str) -> None:
        """
        Callback for when rephrasing from the floating window fails.

        Args:
            error_message (str): The error message from the worker.
        """
        logging.error(f"Post-rephrasing from floating window failed: {error_message}")
        if "empty text" in error_message:
            self.show_tray_balloon(self.translator.tr("rephrasing_failed_empty_message"), 3000)
        else:
            self.show_tray_balloon(self.translator.tr("rephrasing_failed_message", error=error_message), 3000)

    def _normalize_post_rp_entry(self, entry: Dict[str, Any]) -> Dict[str, str]:
        """Return a transformation entry with a stable internal ID for UI reordering."""
        normalized = dict(entry) if isinstance(entry, dict) else {}
        normalized["_entry_id"] = str(normalized.get("_entry_id") or uuid.uuid4().hex)
        normalized["caption"] = str(normalized.get("caption", ""))
        normalized["text"] = str(normalized.get("text", ""))
        return normalized

    def _serialize_post_rp_entries(self) -> List[Dict[str, str]]:
        """Return transformation entries without internal UI metadata for config/runtime use."""
        return [
            {
                "caption": str(entry.get("caption", "")),
                "text": str(entry.get("text", "")),
            }
            for entry in self.post_rephrasing_data
        ]

    def _sync_config_post_rp_entries(self) -> None:
        """Keep the public config representation in sync without exposing internal IDs."""
        self.config["post_rephrasing_entries"] = self._serialize_post_rp_entries()

    def _find_post_rp_entry_by_id(self, entry_id: Optional[str]) -> Optional[Dict[str, str]]:
        """Return the transformation entry for a stable internal entry ID."""
        if not entry_id:
            return None
        for entry in self.post_rephrasing_data:
            if str(entry.get("_entry_id", "")) == entry_id:
                return entry
        return None

    def _load_post_rp_entries_into_list(self) -> None:
        """(Re)loads all entries from the data model into the list widget."""
        self.post_rp_list.blockSignals(True)
        self.post_rp_list.clear()
        for entry in self.post_rephrasing_data:
            caption = entry.get("caption", "").strip() or self.translator.tr("caption_placeholder")
            item = QListWidgetItem(caption)
            item.setData(Qt.ItemDataRole.UserRole, str(entry.get("_entry_id", "")))
            self.post_rp_list.addItem(item)
        self.post_rp_list.blockSignals(False)
        self._update_post_rp_ui_state()

    def _sync_post_rp_data_from_list(self) -> None:
        """Rebuilds the data model list based on the visual order in the QListWidget after a drag-and-drop."""
        # First, save any uncommitted changes from the editor.
        self._save_pr_editor_changes()

        # Rebuild the data list from the new visual order.
        new_data_list: List[Dict[str, str]] = []
        entry_by_id = {
            str(entry.get("_entry_id", "")): entry
            for entry in self.post_rephrasing_data
        }
        for i in range(self.post_rp_list.count()):
            item = self.post_rp_list.item(i)
            entry_id = str(item.data(Qt.ItemDataRole.UserRole) or "")
            entry = entry_by_id.get(entry_id)
            if entry:
                new_data_list.append(entry)

        # Replace the old list with the newly ordered one.
        self.post_rephrasing_data.clear()
        self.post_rephrasing_data.extend(new_data_list)
        self._sync_config_post_rp_entries()

        # The selection index is the same, but the item at that index is different.
        # We must reload the editor to reflect the item that is now at the selected row.
        self._load_pr_editor_for_row(self.post_rp_list.currentRow())

    def _get_post_rp_entry_for_row(self, row: int) -> Optional[Dict[str, str]]:
        """Return the entry object currently represented by a visual list row."""
        if row < 0:
            return None

        item = self.post_rp_list.item(row)
        if item:
            entry = self._find_post_rp_entry_by_id(str(item.data(Qt.ItemDataRole.UserRole) or ""))
            if entry:
                return entry

        if row < len(self.post_rephrasing_data):
            entry = self.post_rephrasing_data[row]
            if isinstance(entry, dict):
                return entry

        return None

    def _find_post_rp_row_for_entry_id(self, entry_id: Optional[str]) -> int:
        """Return the current visual row for an internal transformation entry ID."""
        if not entry_id:
            return -1
        for row in range(self.post_rp_list.count()):
            item = self.post_rp_list.item(row)
            if item and str(item.data(Qt.ItemDataRole.UserRole) or "") == entry_id:
                return row
        return -1

    def _save_pr_editor_changes(self) -> None:
        """Saves the current editor contents to the data model for the last selected row."""
        if self._post_rp_updating or not self._current_pr_entry_id:
            return

        # Save data via the stable entry ID so reordering cannot redirect the edit
        # into a different row.
        entry = self._find_post_rp_entry_by_id(self._current_pr_entry_id)
        if not entry:
            return
        entry["caption"] = self.post_rp_caption_edit.text()
        entry["text"] = self.post_rp_text_edit.toPlainText()
        self._sync_config_post_rp_entries()

        # Update the matching visual list item, regardless of where it was moved to.
        item = self.post_rp_list.item(self._find_post_rp_row_for_entry_id(self._current_pr_entry_id))
        if item:
            item.setText(entry["caption"].strip() or self.translator.tr("caption_placeholder"))

    def _load_pr_editor_for_row(self, row: int) -> None:
        """Loads data for a given row into the editor fields and updates state."""
        self._post_rp_updating = True
        entry = self._get_post_rp_entry_for_row(row)

        if entry is None:
            # No valid selection, clear and disable editors
            self.post_rp_caption_edit.clear()
            self.post_rp_text_edit.clear()
            self.post_rp_caption_edit.setEnabled(False)
            self.post_rp_text_edit.setEnabled(False)
            self._current_pr_entry_id = None
        else:
            # Valid selection, load data from model into editors
            self.post_rp_caption_edit.setEnabled(True)
            self.post_rp_text_edit.setEnabled(True)
            self.post_rp_caption_edit.setText(entry.get("caption", ""))
            self.post_rp_text_edit.setPlainText(entry.get("text", ""))
            # NOTE: deliberately do NOT steal focus here. This runs on every selection change,
            # so focusing the caption edit would break keyboard navigation: after one Down press
            # focus would jump into the editor and further arrow keys would move the text cursor
            # instead of the list. Focusing the caption on "add new entry" is done there explicitly.
            self._current_pr_entry_id = str(entry.get("_entry_id", ""))

        self._post_rp_updating = False
        self._current_pr_row = row
        self._update_post_rp_ui_state()

    def _on_post_rp_selection_changed(self, current_row: int) -> None:
        """Handles selection changes in the list. Saves old, loads new."""
        if self._post_rp_updating:
            return

        # Save any changes from the previously selected item
        self._save_pr_editor_changes()

        # Load the data for the newly selected item
        self._load_pr_editor_for_row(current_row)

    def _on_post_rp_add_clicked(self) -> None:
        """Adds a new, blank entry."""
        if len(self.post_rephrasing_data) >= self.max_post_rephrasing_entries:
            return
        # Save any pending edits from the current item first
        self._save_pr_editor_changes()

        # Add new entry to the data model
        new_entry = self._normalize_post_rp_entry({"caption": "", "text": ""})
        self.post_rephrasing_data.append(new_entry)
        self._sync_config_post_rp_entries()

        # Re-populate the visual list and select the new item
        self._load_post_rp_entries_into_list()
        self.post_rp_list.setCurrentRow(len(self.post_rephrasing_data) - 1)
        # A fresh entry has an empty caption, so focus the caption edit for immediate typing.
        self.post_rp_caption_edit.setFocus()
        self._update_rephrase_api_group_style() # Update style after adding

    def _on_post_rp_remove_clicked(self) -> None:
        """Removes the currently selected entry."""
        row = self.post_rp_list.currentRow()
        entry = self._get_post_rp_entry_for_row(row)
        if entry:
            self.post_rephrasing_data = [
                existing_entry
                for existing_entry in self.post_rephrasing_data
                if str(existing_entry.get("_entry_id", "")) != str(entry.get("_entry_id", ""))
            ]
            self._sync_config_post_rp_entries()

            # Determine which row to select next
            new_row = row
            if new_row >= len(self.post_rephrasing_data):
                new_row = len(self.post_rephrasing_data) - 1

            # Invalidate current row before reloading everything
            self._current_pr_row = -1
            self._current_pr_entry_id = None
            self._load_post_rp_entries_into_list()

            if new_row >= 0:
                self.post_rp_list.setCurrentRow(new_row)
            else:
                # The list is now empty, clear the editor
                self._load_pr_editor_for_row(-1)
            self._update_rephrase_api_group_style() # Update style after removing

    def _update_post_rp_ui_state(self) -> None:
        """Enables/disables add/remove buttons based on item count."""
        count = len(self.post_rephrasing_data)
        self.post_rp_add_btn.setEnabled(count < self.max_post_rephrasing_entries)
        self.post_rp_remove_btn.setEnabled(count > 0 and self.post_rp_list.currentRow() >= 0)

    def _save_current_post_rp_edits(self) -> None:
        """Explicitly saves the currently visible editor state to the data model."""
        self._save_pr_editor_changes()
