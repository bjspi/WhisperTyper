"""ClipboardMixin — clipboard capture/restore and text insertion."""
from __future__ import annotations

import logging
import subprocess
import time
from typing import Any, Dict, List, Optional

import copykitten
import pyautogui
from pynput import keyboard
from PyQt6.QtCore import QByteArray, QMimeData
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication

from app.core.env import is_MACOS, is_WINDOWS
from app.core.timing import OperationTiming
from app.services.windows_text_input import send_unicode_text


class ClipboardMixin:
    """Clipboard capture/restore and text insertion."""

    _pending_clipboard_restore_state: Optional[Dict[str, Any]]

    def _simulate_key_combination(self, char: str) -> bool:
        """
        Simulates pressing a key combination like Ctrl+C or Cmd+C.
        Uses either pynput or pyautogui based on user configuration.

        Args:
            char (str): The character key to press (e.g., 'c', 'v').
        """
        if is_MACOS:
            try:
                # On macOS, sending Command combinations through System Events is
                # more reliable than synthesized key-down/key-up sequences.
                script = f'tell application "System Events" to keystroke "{char}" using command down'
                subprocess.run(['osascript', '-e', script], check=True, capture_output=True, text=True)
                return True
            except subprocess.CalledProcessError as e:
                logging.warning(
                    "macOS System Events key simulation failed for Cmd+%s: %s",
                    char,
                    (e.stderr or e.stdout or str(e)).strip()
                )
            except Exception as e:
                logging.warning(f"macOS System Events key simulation failed for Cmd+{char}: {e}")

        use_alt_lib = self.config.get("alt_clipboard_lib", False)

        if use_alt_lib:
            logging.debug("Using pyautogui for key simulation.")
            modifier = 'command' if is_MACOS else 'ctrl'
            try:
                if (is_WINDOWS or is_MACOS) and char == 'v' and self.config.get("windows_fast_paste", False):
                    pyautogui.hotkey(modifier, char, _pause=False)
                else:
                    pyautogui.hotkey(modifier, char)
                return True
            except Exception as e:
                logging.error(f"pyautogui key simulation failed: {e}")
        else:
            logging.debug("Using pynput for key simulation.")
            modifier = keyboard.Key.cmd if is_MACOS else keyboard.Key.ctrl
            try:
                with self.keyboard_controller.pressed(modifier):
                    self.keyboard_controller.press(char)
                    self.keyboard_controller.release(char)
                return True
            except Exception as e:
                logging.error(f"pynput key simulation failed: {e}")
        return False

    def _capture_clipboard_state(self) -> Dict[str, Any]:
        """Capture the clipboard state before temporary copy/paste operations."""
        if self._pending_clipboard_restore_state:
            # If a paste-triggered restore is still pending, preserve
            # the original user clipboard content instead of the app's temporary text.
            logging.debug("Using pending clipboard restore snapshot as the current clipboard baseline.")
            return self._pending_clipboard_restore_state

        state: Dict[str, Any] = {
            "mode": "text",
            "text": "",
            "has_text": False,
        }

        # Qt exposes native clipboard formats on Windows and macOS (including custom
        # binary formats).  Keep a complete snapshot rather than reducing the user's
        # clipboard to the plain-text representation returned by copykitten.
        mime_snapshot = self._capture_qt_clipboard_state()
        if mime_snapshot is not None:
            state["mode"] = "qt_mime"
            state["mime_snapshot"] = mime_snapshot
            try:
                state["text"] = copykitten.paste()
                state["has_text"] = True
            except Exception:
                pass
            return state

        try:
            state["text"] = copykitten.paste()
            state["has_text"] = True
        except Exception:
            logging.warning("Could not read initial clipboard text state for restoration.")

        return state

    def _restore_clipboard_state(self, state: Dict[str, Any]) -> None:
        """Restore the clipboard state captured before a temporary copy/paste operation."""
        if not state:
            return

        if state.get("mode") in {"qt_mime", "macos_mime"}:
            if self._restore_qt_clipboard_state(state.get("mime_snapshot")):
                logging.debug("Clipboard content restored.")
                return
            logging.warning("Falling back to plain-text clipboard restoration.")

        if state.get("has_text"):
            copykitten.copy(str(state.get("text", "")))
        else:
            copykitten.clear()
        logging.debug("Clipboard content restored.")

    def _capture_qt_clipboard_state(self) -> Optional[Dict[str, Any]]:
        """
        Capture the full Qt/native clipboard payload instead of only plain text.

        Most clipboard values can be recreated from ``QMimeData.data()``. Images are
        special: Qt commonly reports ``application/x-qt-image`` with zero raw bytes and
        keeps the actual pixels in ``imageData()``. Store a detached QImage as well so
        screenshots and other bitmap clipboard contents survive the round-trip.
        """
        try:
            clipboard = QApplication.clipboard()
            mime_data = clipboard.mimeData() if clipboard else None
        except Exception as e:
            logging.warning(f"Could not access Qt clipboard for snapshot: {e}")
            return None

        if mime_data is None:
            return {"formats": []}

        formats_snapshot: List[Dict[str, Any]] = []
        for mime_format in mime_data.formats():
            try:
                formats_snapshot.append({
                    "format": mime_format,
                    "data": mime_data.data(mime_format).data(),
                })
            except Exception as e:
                logging.warning(f"Could not snapshot clipboard format '{mime_format}': {e}")

        snapshot: Dict[str, Any] = {"formats": formats_snapshot}
        if mime_data.hasImage():
            try:
                image_data = mime_data.imageData()
                if isinstance(image_data, QImage):
                    snapshot["image"] = image_data.copy()
                elif isinstance(image_data, QPixmap):
                    snapshot["image"] = image_data.toImage().copy()
                else:
                    logging.warning(
                        "Clipboard advertised image data in unsupported Qt type %s.",
                        type(image_data).__name__,
                    )
            except Exception as e:
                logging.warning(f"Could not snapshot clipboard image data: {e}")

        return snapshot

    def _restore_qt_clipboard_state(self, snapshot: Any) -> bool:
        """
        Restore a previously captured full Qt/native clipboard snapshot.

        The QMimeData object is intentionally handed to QClipboard without a parent;
        Qt takes ownership of it. Raw native formats are restored first and semantic
        image data last, because setting an empty ``application/x-qt-image`` byte array
        after ``setImageData`` would otherwise erase the actual pixels.

        Falls back to text-only restoration if any part of the native MIME restore fails.
        """
        if not isinstance(snapshot, dict):
            return False

        try:
            clipboard = QApplication.clipboard()
        except Exception as e:
            logging.warning(f"Could not access Qt clipboard for restoration: {e}")
            return False

        if clipboard is None:
            return False

        try:
            formats_snapshot = snapshot.get("formats", [])
            image_snapshot = snapshot.get("image")
            if not formats_snapshot and not isinstance(image_snapshot, QImage):
                clipboard.clear()
                QApplication.processEvents()
                return True

            restored_mime = QMimeData()
            for format_entry in formats_snapshot:
                mime_format = str(format_entry.get("format", ""))
                if not mime_format:
                    continue
                mime_bytes = format_entry.get("data", b"")
                if not isinstance(mime_bytes, (bytes, bytearray)):
                    mime_bytes = bytes(mime_bytes)
                restored_mime.setData(mime_format, QByteArray(bytes(mime_bytes)))

            if isinstance(image_snapshot, QImage):
                restored_mime.setImageData(image_snapshot.copy())

            clipboard.setMimeData(restored_mime)
            QApplication.processEvents()
            return True
        except Exception as e:
            logging.warning(f"Failed to restore Qt clipboard MIME snapshot: {e}")
            return False

    def _schedule_clipboard_restore(self, state: Dict[str, Any], delay_ms: int = 500) -> None:
        """
        Delay restoration until the target application has consumed the paste event.

        Sending Ctrl/Cmd+V only queues keyboard input in the target process. Applications
        such as Notepad++ may not execute their paste handler until well after key
        simulation returns, so restoring synchronously can replace the temporary text
        before the target reads it.
        """
        self._pending_clipboard_restore_state = state
        self._clipboard_restore_timer.stop()
        self._clipboard_restore_timer.start(max(0, delay_ms))
        logging.debug("Scheduled clipboard restore %sms after paste.", delay_ms)

    def _perform_clipboard_restore(self) -> None:
        """Restore the delayed clipboard snapshot after the target paste has settled."""
        state = self._pending_clipboard_restore_state
        self._pending_clipboard_restore_state = None
        if not state:
            return
        try:
            self._restore_clipboard_state(state)
        except Exception as e:
            logging.error(f"Failed to restore delayed macOS clipboard state: {e}")

    def get_selected_text(self, select_all_first: bool = False) -> str:
        """
        Retrieves the currently selected text from any application by copying it to the clipboard.
        This method temporarily clears the clipboard to ensure that it captures the new selection,
        even if the selected text was already in the clipboard.

        Restores original clipboard content if configured.

        Args:
            select_all_first (bool): Whether to trigger a platform-aware "select all"
                before copying the text from the focused field.

        Returns:
            str: The selected text, or an empty string if nothing is selected or an error occurs.
        """
        self._check_and_warn_macos_permissions('accessibility')
        selected_text = ""
        restore = self.config["restore_clipboard"]
        original_clipboard_state: Dict[str, Any] = {}

        try:
            # 1. Store original clipboard content if it needs to be restored.
            if restore:
                original_clipboard_state = self._capture_clipboard_state()

            # 2. Optionally select all text in the currently focused field first.
            if select_all_first:
                logging.debug("Selecting all text in the focused field before copying selection.")
                self._simulate_key_combination('a')
                time.sleep(0.08)

            # 3. Clear the clipboard to reliably detect if the copy command succeeds.
            copykitten.copy("")

            # 4. Simulate Ctrl+C / Cmd+C to copy selected text.
            self._simulate_key_combination('c')

            # 5. Wait a moment for the OS to process the copy command.
            time.sleep(0.1)

            # 6. Get the new clipboard content.
            selected_text = copykitten.paste()

            if not selected_text:
                logging.debug("No text selected (clipboard is empty after copy action).")

        except Exception as e:
            logging.error(f"Failed to retrieve selected text: {e}")
            selected_text = "" # Ensure it's empty on error
        finally:
            # 7. Restore the original clipboard content if the setting is enabled.
            if restore:
                try:
                    self._restore_clipboard_state(original_clipboard_state)
                except Exception as e:
                    logging.error(f"Failed to restore clipboard: {e}")

        return selected_text.strip()

    def insert_transcribed_text(self, text: str, timing: Optional[OperationTiming] = None) -> bool:
        """
        Inserts text using optional Windows Unicode input or the existing clipboard path.

        On the clipboard path, 'Restore clipboard' saves and restores the original content.
        If disabled, the new text remains on the clipboard after pasting.

        Args:
            text (str): The text to insert.
            timing: Optional operation milestones; no synchronous log writes.

        Returns:
            Whether input dispatch succeeded; the target application does not acknowledge it.
        """
        if not text:
            return True

        if timing:
            timing.mark("text_commit_start")

        if is_WINDOWS and self.config.get("windows_sendinput_text", False):
            if timing:
                timing.mark("sendinput_dispatch_start")
            accepted, expected = 0, 0
            can_fallback = True
            try:
                accepted, expected = send_unicode_text(text)
            except (OSError, ValueError) as error:
                logging.warning("text_input mode=sendinput rejected error=%s", type(error).__name__)
            except Exception as error:
                can_fallback = False
                logging.error("text_input mode=sendinput unknown_failure error=%s", type(error).__name__)
            if timing:
                timing.mark("sendinput_dispatch_end")
            logging.info(
                "text_input op=%s mode=sendinput accepted_events=%s total_events=%s",
                timing.operation_id if timing else "none", accepted, expected,
            )
            if expected and accepted == expected:
                if timing:
                    timing.mark("text_commit_end")
                return True
            if accepted or not can_fallback or not self.config.get("windows_sendinput_fallback", True):
                if timing:
                    timing.mark("text_commit_failed")
                self.show_tray_balloon(self.translator.tr("windows_sendinput_failed_message"), 4500)
                return False
            if timing:
                timing.mark("sendinput_fallback")
            logging.info("text_input op=%s mode=clipboard fallback=sendinput_rejected", timing.operation_id if timing else "none")

        # Ensure the user is prompted for permissions on macOS before trying to paste.
        self._check_and_warn_macos_permissions('accessibility')

        logging.debug("Inserting transcribed text via clipboard paste for reliability.")
        fast_paste = (is_WINDOWS or is_MACOS) and self.config.get("windows_fast_paste", False)
        logging.info("text_input op=%s mode=clipboard fast_paste=%s", timing.operation_id if timing else "none", fast_paste)
        restore = self.config["restore_clipboard"]
        old_clipboard_state: Dict[str, Any] = {}
        try:
            if restore:
                if timing:
                    timing.mark("clipboard_snapshot_start")
                old_clipboard_state = self._capture_clipboard_state()
                if timing:
                    timing.mark("clipboard_snapshot_end")

            # Copy the new text to the clipboard. This is necessary for special characters.
            if timing:
                timing.mark("clipboard_write_start")
            copykitten.copy(text)
            if timing:
                timing.mark("clipboard_write_end")

            # Wait a moment to ensure the OS has processed the copy command.
            if not fast_paste:
                time.sleep(0.1)

            # Platform-aware paste hotkey
            if timing:
                timing.mark("paste_dispatch_start")
            if not self._simulate_key_combination('v'):
                if timing:
                    timing.mark("text_commit_failed")
                return False
            if timing:
                timing.mark("paste_dispatch_end")
                timing.mark("text_commit_end")

            # Give the target application a moment to process the paste command.
            if not fast_paste:
                time.sleep(0.1)
            if timing:
                timing.mark("paste_settle_end")
            return True

        except Exception as e:
            if timing:
                timing.mark("text_commit_failed")
            logging.error(f"Failed to insert text via clipboard: {e}")
            return False
        finally:
            # Once the original state was captured, always put it back—even if copying
            # or simulating the paste fails halfway through the operation.
            if restore and old_clipboard_state:
                try:
                    # Keep ~600 ms after dispatch without blocking the GUI for the first 100 ms.
                    self._schedule_clipboard_restore(old_clipboard_state, delay_ms=600 if fast_paste else 500)
                except Exception as e:
                    logging.error(f"Failed to restore clipboard after insertion: {e}")

    def copy_last_transcription_to_clipboard(self) -> None:
        """Copies the last transcription to the system clipboard."""
        if not self.last_transcription:
            self.show_tray_balloon(self.translator.tr("no_transcription_to_copy_message"), 2000)
            return
        copykitten.copy(self.last_transcription)
        self.show_tray_balloon(self.translator.tr("transcription_copied_message"), 2000)
