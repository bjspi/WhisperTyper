"""Global hotkeys: listener lifecycle, press/release dispatch and the "Set hotkey" capture flow.

Pure token/binding logic (normalization, parsing, VK maps, matching) lives in
``app.core.hotkeys`` and key-event conversion in ``app.hotkeys.key_tokens``.

Threading contract: pynput and Win32 callbacks run on the listeners' own threads. They never
touch Qt widgets; detected actions leave through ``action_triggered`` and capture previews
through queued signals, so every GUI update happens on the GUI thread.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Protocol, Set

from pynput import keyboard
from PyQt6.QtCore import QEvent, QObject, Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QLineEdit, QPushButton

from app.context import AppContext
from app.controllers.permissions import MacPermissions
from app.core import hotkeys
from app.core.env import is_MACOS, is_WINDOWS
from app.hotkeys.key_tokens import injected_event_counts, pynput_key_tokens, qt_key_tokens
from app.hotkeys.windows_listener import WindowsHotkeyListener
from app.ui.durations import BALLOON_WARNING_MS

#: macOS: an armed post-rephrase whose keys are released later than this is dropped.
_RELEASE_ACTION_TIMEOUT_S = 3.0


class RecordingState(Protocol):
    """What the listeners need to know about the recording (read and written from listener threads)."""

    is_recording: bool
    push_to_talk_active: bool


class HotkeyController(QObject):
    """Run the global hotkey listeners and report detected actions with their detection time."""

    #: ``(action, detected_ns)``; emitted on listener threads, so connections are queued.
    action_triggered = pyqtSignal(str, object)
    _capture_text = pyqtSignal(str)
    _capture_finished = pyqtSignal()

    def __init__(self, ctx: AppContext, recording: RecordingState, permissions: MacPermissions) -> None:
        """``recording`` exposes ``is_recording`` and the shared ``push_to_talk_active`` flag."""
        super().__init__()
        self._ctx = ctx
        self._config = ctx.config
        self._recording = recording
        self._permissions = permissions
        self.bindings: List[Dict[str, Any]] = []
        self._manual_bindings: List[Dict[str, Any]] = []
        self._pressed: Set[str] = set()
        self._active_actions: Set[str] = set()
        #: macOS: actions armed on press (with arm time) that fire once their keys are released.
        self._release_actions: Dict[str, float] = {}
        #: macOS: global callbacks stay muted until this time after a capture (monotonic seconds).
        self._muted_until = 0.0
        self._manual_listener: Optional[keyboard.Listener] = None
        self._windows_listener: Optional[WindowsHotkeyListener] = None
        self._capture_listener: Optional[keyboard.Listener] = None
        self._capture_widget: Optional[QLineEdit] = None
        self._capture_button: Optional[QPushButton] = None
        self._captured_keys: Set[str] = set()
        self._capture_text.connect(self._apply_captured_text)
        self._capture_finished.connect(self.finish_capture)

    # --- Listeners -----------------------------------------------------------------------
    def stop(self) -> None:
        """Stop all active hotkey listeners."""
        if self._manual_listener and self._manual_listener.is_alive():
            self._manual_listener.stop()
        self._manual_listener = None

        if self._windows_listener and self._windows_listener.is_alive():
            self._windows_listener.stop()
            self._windows_listener.join(timeout=1.0)
        self._windows_listener = None

    def restart(self) -> None:
        """(Re)build the bindings from the config and start the listeners."""
        self.stop()
        # Reassign (never mutate in place): the previous set/list objects may still be read
        # by a listener thread that is just shutting down; fresh objects make the swap atomic.
        self._pressed = set()
        self._active_actions = set()
        self._release_actions = {}
        self._recording.push_to_talk_active = False

        if is_MACOS:
            self._permissions.ensure_hotkey_permissions()

        hotkey_str = self._config["hotkey"]
        post_rephrase_hotkey_str = self._config["post_rephrase_hotkey"]
        bindings: List[Dict[str, Any]] = []
        main_binding = hotkeys.parse_hotkey_binding(hotkey_str, "transcription")
        post_binding = hotkeys.parse_hotkey_binding(post_rephrase_hotkey_str, "post_rephrase")
        if main_binding:
            bindings.append(main_binding)
        if post_binding:
            bindings.append(post_binding)

        push_to_talk = self._config.get("push_to_talk", False)

        def _use_windows_registration(binding: Dict[str, Any]) -> bool:
            """OS-level RegisterHotKey when possible; the pynput hook covers the rest."""
            return (
                is_WINDOWS
                and binding["windows_bindable"]
                and not push_to_talk
                and not hotkeys.binding_needs_manual_suppression(binding)
            )

        self.bindings = bindings
        self._manual_bindings = [b for b in bindings if not _use_windows_registration(b)]
        windows_bindings = [
            {**binding, "id": index}
            for index, binding in enumerate(bindings, start=1)
            if _use_windows_registration(binding)
        ]

        if not self.bindings:
            logging.warning("No valid hotkeys set. Hotkey listener will not start.")
            return

        if windows_bindings:
            self._windows_listener = WindowsHotkeyListener(
                windows_bindings,
                self.action_triggered.emit,
                on_registration_failed=self._on_windows_registration_failed,
            )
            self._windows_listener.start()

        if self._manual_bindings:
            self._manual_listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
                win32_event_filter=self._win32_event_filter if is_WINDOWS else None
            )
            try:
                self._manual_listener.start()
            except Exception as e:
                logging.warning(f"Failed to start manual hotkey listener: {e}")
                self._manual_listener = None
                if is_MACOS:
                    self._permissions.warn('input_monitoring')
                    self._permissions.warn('accessibility')

        logging.info(f"Hotkey listeners started for combos: '{hotkey_str}' and '{post_rephrase_hotkey_str}'")

    def _binding_for(self, action: str) -> Optional[Dict[str, Any]]:
        """Return the configured binding for a given action, if any."""
        return next((binding for binding in self.bindings if binding["action"] == action), None)

    def _should_suppress_manual_windows_event(self, key_tokens: Set[str], is_press: bool) -> bool:
        """Return whether the current manual Windows event should be suppressed."""
        if not key_tokens:
            return False

        if is_press:
            projected_tokens = self._pressed.union(key_tokens)
            for binding in self._manual_bindings:
                if not hotkeys.binding_needs_manual_suppression(binding):
                    continue
                if (
                    "<caps_lock>" in key_tokens
                    and "<caps_lock>" in binding["tokens"]
                    and projected_tokens.issubset(binding["tokens"])
                    and bool((projected_tokens - {"<caps_lock>"}).intersection(binding["tokens"]))
                ):
                    return True
                if binding["modifiers"].issubset(projected_tokens):
                    trigger_tokens = binding["trigger_tokens"] or binding["tokens"]
                    if key_tokens.intersection(trigger_tokens):
                        return True
            return False

        return self._is_push_to_talk_release(key_tokens)

    def _is_push_to_talk_release(self, released_tokens: Set[str]) -> bool:
        """Whether ``released_tokens`` end the push-to-talk recording that is running now."""
        transcription_binding = self._binding_for("transcription")
        return bool(
            self._config.get("push_to_talk", False)
            and self._recording.push_to_talk_active
            and self._recording.is_recording
            and transcription_binding
            and released_tokens.intersection(hotkeys.binding_release_tokens(transcription_binding))
        )

    def _win32_event_filter(self, msg: int, data: Any) -> bool:
        """Suppress Windows hotkey key events in-hook so they do not reach the active application."""
        listener = self._manual_listener
        key_tokens = hotkeys.vk_to_hotkey_tokens(getattr(data, "vkCode", None))
        is_press = msg in (0x0100, 0x0104)  # WM_KEYDOWN / WM_SYSKEYDOWN
        is_release = msg in (0x0101, 0x0105)  # WM_KEYUP / WM_SYSKEYUP
        if listener:
            listener._suppress = bool(key_tokens and (is_press or is_release)
                                      and self._should_suppress_manual_windows_event(key_tokens, is_press))
        return True

    def _on_press(self, key: Any, injected: bool = False) -> None:
        """Pynput callback (listener thread) for any key press."""
        detected_ns = time.perf_counter_ns()
        key_tokens = pynput_key_tokens(key)
        if injected and not injected_event_counts(key_tokens):
            return
        if is_MACOS and (self.capturing or time.monotonic() < self._muted_until):
            return  # macOS keeps the listener running during capture (see start_capture)
        self._pressed.update(key_tokens)

        # Snapshot the binding list: the main thread swaps in a new list on re-init.
        for binding in list(self._manual_bindings):
            if (hotkeys.binding_matches_current_press(binding, self._pressed, key_tokens)
                    and binding["action"] not in self._active_actions):
                self._active_actions.add(binding["action"])
                logging.info(f"Manual hotkey combo detected: {binding['display']}")
                if is_MACOS and binding["action"] == "post_rephrase":
                    # Copying the selection while e.g. Ctrl is still held would send Ctrl+Cmd+C,
                    # which apps ignore, so this action waits for the release of its keys.
                    self._release_actions[binding["action"]] = time.monotonic()
                elif binding["action"] == "transcription" and self._config.get("push_to_talk", False):
                    if not self._recording.is_recording:
                        self._recording.push_to_talk_active = True
                        self.action_triggered.emit(binding["action"], detected_ns)
                else:
                    self.action_triggered.emit(binding["action"], detected_ns)
                if binding["action"] == "transcription":
                    return

    def _on_release(self, key: Any, injected: bool = False) -> None:
        """Pynput callback (listener thread) for any key release."""
        detected_ns = time.perf_counter_ns()
        released_tokens = pynput_key_tokens(key)
        if injected and not injected_event_counts(released_tokens):
            return
        if is_MACOS and (self.capturing or time.monotonic() < self._muted_until):
            return
        if self._is_push_to_talk_release(released_tokens):
            logging.info("Push-to-talk hotkey released. Stopping recording.")
            self._recording.push_to_talk_active = False
            self.action_triggered.emit("stop_transcription", detected_ns)

        self._pressed.difference_update(released_tokens)

        for binding in list(self._manual_bindings):
            armed_at = self._release_actions.get(binding["action"])
            if armed_at is not None and not binding["tokens"] & self._pressed:
                del self._release_actions[binding["action"]]
                # A release that only arrives much later (e.g. a lost key-up) must not act on a new selection.
                if time.monotonic() - armed_at < _RELEASE_ACTION_TIMEOUT_S:
                    self.action_triggered.emit(binding["action"], detected_ns)

        still_active = {
            binding["action"]
            for binding in list(self._manual_bindings)
            if hotkeys.binding_matches_pressed(binding, self._pressed)
        }
        self._active_actions.intersection_update(still_active)

    def _on_windows_registration_failed(self, display: str) -> None:
        """Surface a Windows hotkey that could not be registered (listener thread; the notifier is thread-safe)."""
        logging.warning(f"Windows hotkey {display} could not be registered; it will not fire.")
        try:
            self._ctx.notifier.show(self._ctx.tr("hotkey_register_failed_message", hotkey=display), BALLOON_WARNING_MS)
        except Exception:
            pass

    # --- "Set hotkey" capture flow -------------------------------------------------------
    @property
    def capturing(self) -> bool:
        """Whether a capture is in progress."""
        return self._capture_widget is not None or self._capture_button is not None

    def start_capture(self, target: QLineEdit, button: QPushButton) -> None:
        """Record the next key combination into ``target``; ``button`` shows the listening state."""
        self._permissions.warn('input_monitoring')
        self._capture_widget = target
        self._capture_button = button
        button.setText(self._ctx.tr("hotkey_listening_button"))
        button.setEnabled(False)
        self._captured_keys = set()

        if is_MACOS:
            # Stopping the CoreGraphics listener from this Qt callback aborts the process on recent
            # macOS (SIGTRAP in Text Input Services). It keeps running; its callbacks ignore keys
            # while ``capturing`` is set.
            target.clear()
            target.installEventFilter(self)
            target.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
            target.grabKeyboard()
            return

        # Suspend the GLOBAL hotkeys while capturing. Otherwise pressing e.g. F9 both records the
        # key AND fires its global action (post-rephrase), whose simulated Ctrl+C then gets caught
        # by the capture listener too — producing garbage like "<ctrl>++<f9>+c".
        self.stop()
        self._capture_listener = keyboard.Listener(on_press=self._on_capture_press,
                                                   on_release=self._on_capture_release)
        self._capture_listener.start()

    def eventFilter(self, watched: Optional[QObject], event: Optional[QEvent]) -> bool:
        """Capture hotkeys in the macOS settings UI without relying on pynput capture."""
        if is_MACOS and event is not None and watched is not None and watched == self._capture_widget:
            if event.type() == QEvent.Type.ShortcutOverride:
                return True
            if event.type() == QEvent.Type.KeyPress and isinstance(event, QKeyEvent):
                tokens = qt_key_tokens(event.key(), event.modifiers(), event.text())
                if tokens:
                    self._capture_widget.setText(hotkeys.format_hotkey_tokens(tokens))
                    self._capture_widget.selectAll()
                    if any(not hotkeys.is_modifier_token(token) for token in tokens):
                        self.finish_capture()
                return True
        return super().eventFilter(watched, event)

    def _on_capture_press(self, key: Any) -> None:
        """Pynput capture callback (listener thread): record the key, preview via signal."""
        if not self._capture_widget:
            return
        self._captured_keys.update(pynput_key_tokens(key))
        self._capture_text.emit(hotkeys.format_hotkey_tokens(self._captured_keys))

    def _on_capture_release(self, _key: Any) -> None:
        """Pynput capture callback (listener thread): finalize the capture via signals."""
        if self._capture_listener:
            self._capture_listener.stop()
            self._capture_listener = None

        if self._capture_widget:
            self._capture_text.emit(hotkeys.format_hotkey_tokens(self._captured_keys))
            # finish_capture touches buttons and restarts the listeners: GUI thread only.
            self._capture_finished.emit()

    def _apply_captured_text(self, text: str) -> None:
        """GUI thread: show the current capture preview in the target field."""
        if self._capture_widget:
            self._capture_widget.setText(text)

    def finish_capture(self) -> None:
        """Reset the capture UI and restore the global listeners (GUI thread)."""
        if not self.capturing:
            return  # capture already finished/cancelled; a late signal must not re-run this

        if is_MACOS and self._capture_widget:
            self._capture_widget.releaseKeyboard()
            self._capture_widget.removeEventFilter(self)

        if self._capture_button:
            self._capture_button.setText(self._ctx.tr("set_hotkey_button"))
            self._capture_button.setEnabled(True)

        if is_MACOS:
            # The listener never stopped; skip the keys of the capture that are still in flight.
            # A new hotkey applies after an app restart (see the settings window).
            self._muted_until = time.monotonic() + 0.25
            self._pressed = set()
            self._active_actions = set()
            self._release_actions = {}
        self._capture_widget = None
        self._capture_button = None
        if not is_MACOS:
            self.restart()

    def cancel_capture(self) -> None:
        """Abort an in-progress capture and restore the global listeners.

        ``start_capture`` stops every global listener so the keys pressed for capture are not
        also fired as hotkeys. If the capture is abandoned (e.g. the settings window is closed
        before a key is pressed), this restarts them; otherwise all hotkeys would stay dead.
        """
        if not self.capturing:
            return
        if self._capture_listener:
            try:
                self._capture_listener.stop()
            except Exception:
                pass
            self._capture_listener = None
        logging.info("Hotkey capture aborted; restoring global hotkey listeners.")
        self.finish_capture()
