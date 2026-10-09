"""Text insertion into the focused application and reading its current selection."""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional

import copykitten
from PyQt6.QtCore import QObject, QTimer

from app.core.env import is_MACOS, is_WINDOWS
from app.core.timing import NO_TIMING, OperationTiming
from app.services.clipboard import ClipboardSnapshot
from app.services.key_simulation import simulate_shortcut
from app.services.windows_text_input import send_unicode_text
from app.ui.durations import (
    BALLOON_LONG_MS,
    CLIPBOARD_RESTORE_FAST_MS,
    CLIPBOARD_RESTORE_MS,
    COPY_SETTLE_S,
    PASTE_PREPARE_S,
    PASTE_SETTLE_S,
    SELECT_ALL_SETTLE_S,
)


class TextOutput(QObject):
    """Insert text via SendInput or clipboard paste; read the selection via a simulated copy."""

    def __init__(self, config: Dict[str, Any], tr: Callable[..., str], notify: Callable[..., None],
                 warn_permission: Callable[[str], None]) -> None:
        """``notify`` shows a balloon; ``warn_permission`` explains a missing macOS permission."""
        super().__init__()
        self._config = config
        self._tr = tr
        self._notify = notify
        self._warn_permission = warn_permission
        self._pending_restore: Optional[ClipboardSnapshot] = None
        self._restore_timer: Any = QTimer(self)
        self._restore_timer.setSingleShot(True)
        self._restore_timer.timeout.connect(self.restore_clipboard_now)

    def _simulate_key_combination(self, char: str) -> bool:
        """Press Ctrl/Cmd+``char`` with the configured key library; fast paste skips its pause."""
        return simulate_shortcut(char, alt_lib=bool(self._config.get("alt_clipboard_lib", False)),
                                 fast=char == 'v' and self._fast_paste_enabled())

    def _capture_clipboard_state(self) -> ClipboardSnapshot:
        """Snapshot the user's clipboard before a temporary copy/paste."""
        if self._pending_restore:
            # A paste-triggered restore is still pending: the clipboard holds the app's temporary
            # text, so the pending snapshot is the user's real content.
            logging.debug("Using pending clipboard restore snapshot as the current clipboard baseline.")
            return self._pending_restore
        return ClipboardSnapshot.capture()

    def _schedule_clipboard_restore(self, snapshot: ClipboardSnapshot, delay_ms: int = CLIPBOARD_RESTORE_MS) -> None:
        """
        Delay restoration until the target application has consumed the paste event.

        Sending Ctrl/Cmd+V only queues keyboard input in the target process. Applications
        such as Notepad++ may not execute their paste handler until well after key
        simulation returns, so restoring synchronously can replace the temporary text
        before the target reads it.
        """
        self._pending_restore = snapshot
        self._restore_timer.stop()
        self._restore_timer.start(max(0, delay_ms))
        logging.debug("Scheduled clipboard restore %sms after paste.", delay_ms)

    @property
    def restore_pending(self) -> bool:
        """Whether a delayed clipboard restore is still scheduled."""
        return self._pending_restore is not None

    def restore_clipboard_now(self) -> None:
        """Restore the delayed clipboard snapshot (timer expiry, or immediately at quit)."""
        self._restore_timer.stop()
        snapshot, self._pending_restore = self._pending_restore, None
        if not snapshot:
            return
        try:
            snapshot.restore()
        except Exception as e:
            logging.error(f"Failed to restore delayed clipboard state: {e}")

    def get_selected_text(self, select_all_first: bool = False) -> str:
        """
        Retrieves the currently selected text from any application by copying it to the clipboard.
        The clipboard is cleared first so a selection equal to the old clipboard is still detected;
        the original clipboard is restored if configured.

        Args:
            select_all_first (bool): Whether to trigger a platform-aware "select all"
                before copying the text from the focused field.

        Returns:
            str: The selected text, or an empty string if nothing is selected or an error occurs.
        """
        self._warn_permission('accessibility')
        selected_text = ""
        snapshot: Optional[ClipboardSnapshot] = None
        try:
            if self._config["restore_clipboard"]:
                snapshot = self._capture_clipboard_state()
            if select_all_first:
                logging.debug("Selecting all text in the focused field before copying selection.")
                self._simulate_key_combination('a')
                time.sleep(SELECT_ALL_SETTLE_S)
            copykitten.copy("")
            self._simulate_key_combination('c')
            time.sleep(COPY_SETTLE_S)
            selected_text = copykitten.paste()
            if not selected_text:
                logging.debug("No text selected (clipboard is empty after copy action).")
        except Exception as e:
            logging.error(f"Failed to retrieve selected text: {e}")
            selected_text = ""
        finally:
            if snapshot:
                try:
                    snapshot.restore()
                except Exception as e:
                    logging.error(f"Failed to restore clipboard: {e}")
        return selected_text.strip()

    def _fast_paste_enabled(self) -> bool:
        """Skip the fixed paste waits; offered on Windows and macOS only."""
        return (is_WINDOWS or is_MACOS) and bool(self._config.get("fast_paste", False))

    def insert(self, text: str, timing: OperationTiming = NO_TIMING) -> bool:
        """
        Inserts text using optional Windows Unicode input or the clipboard paste path.

        On the clipboard path, 'Restore clipboard' saves and restores the original content.
        If disabled, the new text remains on the clipboard after pasting.

        Args:
            text (str): The text to insert.
            timing: Operation milestones; no synchronous log writes.

        Returns:
            Whether input dispatch succeeded; the target application does not acknowledge it.
        """
        if not text:
            return True
        timing.mark("text_commit_start")
        if is_WINDOWS and self._config.get("windows_sendinput_text", False):
            result = self._insert_via_sendinput(text, timing)
            if result is not None:
                return result
        return self._insert_via_clipboard(text, timing)

    def _insert_via_sendinput(self, text: str, timing: OperationTiming) -> Optional[bool]:
        """Type via SendInput; None means nothing was accepted and the clipboard may take over."""
        accepted, expected = 0, 0
        can_fallback = True
        with timing.span("sendinput_dispatch"):
            try:
                accepted, expected = send_unicode_text(text)
            except (OSError, ValueError) as error:
                logging.warning("text_input mode=sendinput rejected error=%s", type(error).__name__)
            except Exception as error:
                can_fallback = False
                logging.error("text_input mode=sendinput unknown_failure error=%s", type(error).__name__)
        logging.info("text_input op=%s mode=sendinput accepted_events=%s total_events=%s",
                     timing.operation_id, accepted, expected)
        if expected and accepted == expected:
            timing.mark("text_commit_end")
            return True
        # A partially accepted batch is never retried: the accepted prefix would appear twice.
        if accepted or not can_fallback or not self._config.get("windows_sendinput_fallback", True):
            timing.mark("text_commit_failed")
            self._notify(self._tr("windows_sendinput_failed_message"), BALLOON_LONG_MS)
            return False
        timing.mark("sendinput_fallback")
        logging.info("text_input op=%s mode=clipboard fallback=sendinput_rejected", timing.operation_id)
        return None

    def _insert_via_clipboard(self, text: str, timing: OperationTiming) -> bool:
        """Paste through the clipboard, which handles every character reliably."""
        # Ensure the user is prompted for permissions on macOS before trying to paste.
        self._warn_permission('accessibility')

        fast_paste = self._fast_paste_enabled()
        logging.info("text_input op=%s mode=clipboard fast_paste=%s", timing.operation_id, fast_paste)
        snapshot: Optional[ClipboardSnapshot] = None
        try:
            if self._config["restore_clipboard"]:
                with timing.span("clipboard_snapshot"):
                    snapshot = self._capture_clipboard_state()
            with timing.span("clipboard_write"):
                copykitten.copy(text)
            if not fast_paste:
                time.sleep(PASTE_PREPARE_S)
            timing.mark("paste_dispatch_start")
            if not self._simulate_key_combination('v'):
                timing.mark("text_commit_failed")
                return False
            timing.mark("paste_dispatch_end")
            timing.mark("text_commit_end")
            if not fast_paste:
                time.sleep(PASTE_SETTLE_S)
            timing.mark("paste_settle_end")
            return True
        except Exception as e:
            timing.mark("text_commit_failed")
            logging.error(f"Failed to insert text via clipboard: {e}")
            return False
        finally:
            # Once the original state was captured, always put it back—even if copying
            # or simulating the paste fails halfway through the operation.
            if snapshot:
                try:
                    # Fast paste skips the 100 ms waits, so the restore keeps ~600 ms after dispatch.
                    self._schedule_clipboard_restore(
                        snapshot, CLIPBOARD_RESTORE_FAST_MS if fast_paste else CLIPBOARD_RESTORE_MS)
                except Exception as e:
                    logging.error(f"Failed to restore clipboard after insertion: {e}")
