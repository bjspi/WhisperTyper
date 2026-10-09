"""PostRephraseMixin — rephrase selected text through the floating transformation palette."""
from __future__ import annotations

import logging

import copykitten
from PyQt6.QtWidgets import QWidget

from app.core.api_keys import rephrasing_configured
from app.core.env import is_MACOS
from app.core.prompts import captioned_transformations
from app.core.timing import NO_TIMING, OperationTiming
from app.platform.macos import activate_app, frontmost_app_name
from app.ui.durations import BALLOON_CONFIRM_MS, BALLOON_ERROR_MS, BALLOON_NOTICE_MS, BALLOON_SHORT_MS


class PostRephraseMixin:
    """Post-rephrase transformations editor and trigger."""

    def trigger_post_rephrase_window(self) -> None:
        """Checks for selected text and shows the floating button window if text is present."""
        if not rephrasing_configured(self.config):
            logging.warning("Post-rephrase hotkey pressed, but API settings are missing.")
            self.show_tray_balloon(self.translator.tr("rephrase_api_settings_missing"), BALLOON_ERROR_MS)
            return

        if is_MACOS:
            logging.debug("Trying to get currently active window/application on macOS via osascript")
            try:
                self.macos_active_application = frontmost_app_name()
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
            self.show_tray_balloon(self.translator.tr("no_text_selected_for_rephrase"), BALLOON_SHORT_MS)
            return

        valid_entries = captioned_transformations(self.config.get("post_rephrasing_entries", []))
        if not valid_entries:
            logging.info("Post-rephrase hotkey pressed, but no valid post-processing entries are configured.")
            self.show_tray_balloon(self.translator.tr("no_post_rephrase_entries_configured"), BALLOON_ERROR_MS)
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
        # finished/empty/error callbacks (analogous to the transcription/LivePrompt spinner).
        self.show_tray_balloon(self.translator.tr("rephrasing_selection_message"), 0, spinner=True)

        # Context is not used for this specific action.
        worker = self._build_rephrasing_worker(system_prompt, selected_text)
        self._start_rephrasing_worker(
            worker,
            on_finished=lambda text: self.on_rephrasing_finished(text, worker.timing),
            on_empty=lambda: self._on_rephrasing_failed(self.translator.tr("rephrasing_failed_empty_message"), worker.timing),
            on_error=lambda message: self._on_rephrasing_failed(
                self.translator.tr("rephrasing_failed_message", error=message), worker.timing),
        )

    def on_rephrasing_finished(self, rephrased_text: str, timing: OperationTiming = NO_TIMING) -> None:
        """
        Callback for when rephrasing from the floating window is successful.

        Args:
            rephrased_text (str): The text returned by the AI.
            timing: The operation's request and output timings.
        """
        timing.measure_output(lambda: self._deliver_rephrased_text(rephrased_text, timing))

    def _deliver_rephrased_text(self, rephrased_text: str, timing: OperationTiming) -> bool:
        """Insert into the original application, or copy on macOS when it cannot be reactivated."""
        # On macOS, pasting can be unreliable if the app loses focus.
        # It's safer to copy to clipboard and notify the user.
        if is_MACOS and not self.macos_active_application:
            with timing.span("clipboard_write"):
                copykitten.copy(rephrased_text)
            self.show_tray_balloon(self.translator.tr("rephrasing_finished_macos_message"), BALLOON_NOTICE_MS, check=True)
            return True
        if is_MACOS:
            # Try to reactivate the original app using osascript
            activate_app(self.macos_active_application)
        # Swap the spinner for a brief "done ✓" balloon, then type the text.
        self.show_tray_balloon(self.translator.tr("rephrasing_done_message"), BALLOON_CONFIRM_MS, check=True)
        inserted = self.insert_transcribed_text(rephrased_text, timing=timing)
        if inserted:
            logging.info("Successfully dispatched rephrased text for insertion.")
        return inserted

    def _on_rephrasing_failed(self, notice: str, timing: OperationTiming = NO_TIMING) -> None:
        """Close the operation and replace the spinner with the (translated) failure notice."""
        timing.finish("rephrase_failed")
        logging.error("Post-rephrasing from floating window failed: %s", notice)
        self.show_tray_balloon(notice, BALLOON_ERROR_MS)
