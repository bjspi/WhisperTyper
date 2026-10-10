"""Rephrase selected text through the floating transformation palette."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import copykitten
from PyQt6.QtWidgets import QApplication, QWidget

from app.context import AppContext
from app.controllers.text_output import TextOutput
from app.controllers.transcription import TranscriptionPipeline
from app.core.api_keys import rephrasing_configured
from app.core.env import is_MACOS
from app.core.prompts import captioned_transformations
from app.core.timing import NO_TIMING, OperationTiming
from app.platform.macos import activate_app, frontmost_app_name
from app.ui import theme
from app.ui.durations import BALLOON_CONFIRM_MS, BALLOON_ERROR_MS, BALLOON_NOTICE_MS, BALLOON_SHORT_MS
from app.ui.floating_buttons import FloatingButtonWindow


class PostRephraseController:
    """Read the selection, offer the transformation buttons and replace the selection with the result."""

    def __init__(self, ctx: AppContext, output: TextOutput, pipeline: TranscriptionPipeline) -> None:
        """Rephrasing requests run through the pipeline's shared worker plumbing."""
        self._ctx = ctx
        self._output = output
        self._pipeline = pipeline
        #: Application that had focus when the palette was requested (macOS reactivates it).
        self._macos_active_application: Optional[str] = None

    def trigger(self) -> None:
        """Show the floating transformation buttons for the current selection (GUI thread)."""
        config = self._ctx.config
        if not rephrasing_configured(config):
            logging.warning("Post-rephrase hotkey pressed, but API settings are missing.")
            self._ctx.notifier.show(self._ctx.tr("rephrase_api_settings_missing"), BALLOON_ERROR_MS)
            return

        if is_MACOS:
            logging.debug("Trying to get currently active window/application on macOS via osascript")
            try:
                self._macos_active_application = frontmost_app_name()
            except Exception as e:
                # osascript can fail (e.g. CalledProcessError) when Automation permission is
                # missing; delivery then copies to the clipboard instead.
                logging.warning(f"Could not determine active macOS application: {e}")
                self._macos_active_application = None

        selected_text = self._output.get_selected_text(select_all_first=config.get("post_rephrase_auto_select_all", False))
        if not selected_text:
            logging.info("Post-rephrase hotkey pressed, but no text was selected.")
            self._ctx.notifier.show(self._ctx.tr("no_text_selected_for_rephrase"), BALLOON_SHORT_MS)
            return

        valid_entries = captioned_transformations(config.get("post_rephrasing_entries", []))
        if not valid_entries:
            logging.info("Post-rephrase hotkey pressed, but no valid post-processing entries are configured.")
            self._ctx.notifier.show(self._ctx.tr("no_post_rephrase_entries_configured"), BALLOON_ERROR_MS)
            return
        self.show_palette(valid_entries, selected_text)

    def show_palette(self, entries: List[Dict[str, str]], selected_text: str) -> None:
        """Open the floating window with one button per transformation."""
        FloatingButtonWindow(buttons=entries, selected_text=selected_text,
                             on_button_click_callback=self.on_button_clicked,
                             title=self._ctx.tr("rephrase_window_title"),
                             close_tooltip=self._ctx.tr("rephrase_window_close_tooltip"),
                             dark=theme.resolve_dark(self._ctx.config.get("color_theme", "system"),
                                                     QApplication.instance()))

    def on_button_clicked(self, system_prompt: str, selected_text: str, window: QWidget) -> None:
        """Rephrase ``selected_text`` with the clicked transformation's prompt."""
        window.close()
        logging.info("Floating button clicked. Rephrasing selected text with custom prompt.")
        # Persistent spinner balloon that stays up for the whole request; ended by the
        # finished/empty/error callbacks (analogous to the transcription/LivePrompt spinner).
        self._ctx.notifier.show(self._ctx.tr("rephrasing_selection_message"), 0, spinner=True)
        # Context is not used for this specific action.
        self._pipeline.start_rephrasing(
            system_prompt, selected_text, "", None,
            on_finished=self.on_rephrasing_finished,
            on_empty=lambda timing: self._on_failed(self._ctx.tr("rephrasing_failed_empty_message"), timing),
            on_error=lambda message, timing: self._on_failed(
                self._ctx.tr("rephrasing_failed_message", error=message), timing),
        )

    def on_rephrasing_finished(self, rephrased_text: str, timing: OperationTiming = NO_TIMING) -> None:
        """Deliver the rephrased text and close the operation."""
        timing.measure_output(lambda: self._deliver(rephrased_text, timing))

    def _deliver(self, rephrased_text: str, timing: OperationTiming) -> bool:
        """Insert into the original application, or copy on macOS when it cannot be reactivated."""
        # On macOS, pasting can be unreliable if the app loses focus.
        # It's safer to copy to clipboard and notify the user.
        if is_MACOS and not self._macos_active_application:
            with timing.span("clipboard_write"):
                copykitten.copy(rephrased_text)
            self._ctx.notifier.show(self._ctx.tr("rephrasing_finished_macos_message"), BALLOON_NOTICE_MS, check=True)
            return True
        if is_MACOS and self._macos_active_application:
            activate_app(self._macos_active_application)
        # Swap the spinner for a brief "done ✓" balloon, then type the text.
        self._ctx.notifier.show(self._ctx.tr("rephrasing_done_message"), BALLOON_CONFIRM_MS, check=True)
        inserted = self._output.insert(rephrased_text, timing=timing)
        if inserted:
            logging.info("Successfully dispatched rephrased text for insertion.")
        return inserted

    def _on_failed(self, notice: str, timing: OperationTiming = NO_TIMING) -> None:
        """Close the operation and replace the spinner with the (translated) failure notice."""
        timing.finish("rephrase_failed")
        logging.error("Post-rephrasing from floating window failed: %s", notice)
        self._ctx.notifier.show(notice, BALLOON_ERROR_MS)
