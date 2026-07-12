"""HotkeyMixin — global hotkey listeners, dispatch and capture.

Pure token/binding logic (normalization, parsing, VK maps, matching) lives in
``app.core.hotkeys``; this mixin owns everything that touches real input events:
the pynput/Win32 listener lifecycle, press/release dispatch, and the "Set hotkey"
capture flow.

Threading contract: pynput callbacks run on the listener's own thread. They must never
touch Qt widgets directly — GUI updates are marshalled through the queued signals defined
on the application class (``hotkey_action_signal``, ``hotkey_capture_text_signal``,
``hotkey_capture_finished_signal``).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from pynput import keyboard
from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtGui import QKeyEvent, QWheelEvent

from app.core import hotkeys
from app.core.env import is_MACOS, is_WINDOWS

if is_WINDOWS:
    import ctypes
from app.hotkeys.windows_listener import WindowsHotkeyListener


class HotkeyMixin:
    """Global hotkeys: listeners, press/release dispatch, and capture."""

    def _handle_hotkey_action(self, action: str) -> None:
        """Dispatch hotkey actions onto the Qt main thread."""
        if action == "transcription":
            if self.config.get("push_to_talk", False):
                if not self.is_recording:
                    self.push_to_talk_active = True
                    self.toggle_recording()
            else:
                self.toggle_recording()
        elif action == "stop_transcription":
            if self.is_recording:
                self.push_to_talk_active = False
                self.toggle_recording()
        elif action == "post_rephrase":
            self.trigger_post_rephrase_window()

    def eventFilter(self, watched: QObject, event: Any) -> bool:
        """Capture hotkeys in the macOS settings UI without relying on pynput capture."""
        if is_MACOS and event.type() == QEvent.Type.Wheel and isinstance(event, QWheelEvent):
            if watched in {self.transcription_temp_slider, self.rephrasing_temp_slider}:
                return self._scroll_macos_area_from_slider_wheel(watched, event)

        if is_MACOS and watched == self.capturing_for_widget:
            if event.type() == QEvent.Type.ShortcutOverride:
                return True
            if event.type() == QEvent.Type.KeyPress and isinstance(event, QKeyEvent):
                tokens = self._qt_key_to_hotkey_tokens(event)
                if tokens:
                    self.capturing_for_widget.setText(hotkeys.format_hotkey_tokens(tokens))
                    self.capturing_for_widget.selectAll()
                    non_modifier_tokens = {token for token in tokens if not hotkeys.is_modifier_token(token)}
                    if non_modifier_tokens:
                        self._finish_hotkey_capture()
                return True
        return super().eventFilter(watched, event)

    def _should_suppress_manual_windows_event(self, key_tokens: Set[str], is_press: bool) -> bool:
        """Return whether the current manual Windows event should be suppressed."""
        if not key_tokens:
            return False

        if is_press:
            projected_tokens = self.pressed_hotkey_tokens.union(key_tokens)
            for binding in self.manual_hotkey_bindings:
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

        if self.config.get("push_to_talk", False):
            transcription_binding = self._get_binding_for_action("transcription")
            if (
                self.push_to_talk_active
                and self.is_recording
                and transcription_binding
                and key_tokens.intersection(hotkeys.binding_release_tokens(transcription_binding))
            ):
                return True

        return False

    def _manual_hotkey_win32_event_filter(self, msg: int, data: Any) -> bool:
        """Suppress Windows hotkey key events in-hook so they do not reach the active application."""
        key_tokens = hotkeys.vk_to_hotkey_tokens(getattr(data, "vkCode", None))
        if not key_tokens:
            if self.manual_listener:
                self.manual_listener._suppress = False
            return True

        is_press = msg in (0x0100, 0x0104)  # WM_KEYDOWN / WM_SYSKEYDOWN
        is_release = msg in (0x0101, 0x0105)  # WM_KEYUP / WM_SYSKEYUP
        if not is_press and not is_release:
            if self.manual_listener:
                self.manual_listener._suppress = False
            return True

        if self.manual_listener:
            self.manual_listener._suppress = self._should_suppress_manual_windows_event(key_tokens, is_press)

        return True

    def _stop_hotkey_listeners(self) -> None:
        """Stop all active hotkey listeners before rebuilding them."""
        if self.manual_listener and self.manual_listener.is_alive():
            self.manual_listener.stop()
        self.manual_listener = None

        if self.windows_hotkey_listener and self.windows_hotkey_listener.is_alive():
            self.windows_hotkey_listener.stop()
            self.windows_hotkey_listener.join(timeout=1.0)
        self.windows_hotkey_listener = None

    def _is_console_like_foreground_window(self) -> bool:
        """Return whether the current Windows foreground window is a console/terminal host."""
        if not is_WINDOWS:
            return False

        try:
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            if not hwnd:
                return False

            class_name_buffer = ctypes.create_unicode_buffer(256)
            ctypes.windll.user32.GetClassNameW(hwnd, class_name_buffer, len(class_name_buffer))
            class_name = class_name_buffer.value

            console_classes = {
                "ConsoleWindowClass",
                "CASCADIA_HOSTING_WINDOW_CLASS",
                "VirtualConsoleClass",
                "mintty",
            }
            return class_name in console_classes
        except Exception as e:
            logging.debug(f"Failed to inspect foreground window class: {e}")
            return False

    def init_manual_hotkey_listener(self) -> None:
        """Initializes the manual, low-level keyboard listener."""
        self._stop_hotkey_listeners()
        # Reassign (never mutate in place): the previous set/list objects may still be read
        # by a listener thread that is just shutting down; fresh objects make the swap atomic.
        self.pressed_hotkey_tokens = set()
        self.active_hotkey_actions = set()
        self.push_to_talk_active = False

        if is_MACOS:
            self._ensure_macos_hotkey_permissions()

        bindings: List[Dict[str, Any]] = []
        main_binding = hotkeys.parse_hotkey_binding(self.hotkey_str, "transcription")
        post_binding = hotkeys.parse_hotkey_binding(self.post_rephrase_hotkey_str, "post_rephrase")
        if main_binding:
            bindings.append(main_binding)
        if post_binding:
            bindings.append(post_binding)

        def _use_windows_registration(binding: Dict[str, Any]) -> bool:
            """OS-level RegisterHotKey when possible; the pynput hook covers the rest."""
            return (
                is_WINDOWS
                and binding["windows_bindable"]
                and not self.config.get("push_to_talk", False)
                and not hotkeys.binding_needs_manual_suppression(binding)
            )

        self.hotkey_bindings = bindings
        self.manual_hotkey_bindings = [b for b in bindings if not _use_windows_registration(b)]
        windows_bindings = [
            {**binding, "id": index}
            for index, binding in enumerate(bindings, start=1)
            if _use_windows_registration(binding)
        ]

        if not self.hotkey_bindings:
            logging.warning("No valid hotkeys set. Hotkey listener will not start.")
            return

        if windows_bindings:
            self.windows_hotkey_listener = WindowsHotkeyListener(
                windows_bindings,
                self.hotkey_action_signal.emit,
                on_registration_failed=self._on_windows_hotkey_registration_failed,
            )
            self.windows_hotkey_listener.start()

        if self.manual_hotkey_bindings:
            self.manual_listener = keyboard.Listener(
                on_press=self._on_hotkey_press,
                on_release=self._on_hotkey_release,
                win32_event_filter=self._manual_hotkey_win32_event_filter if is_WINDOWS else None
            )
            try:
                self.manual_listener.start()
            except Exception as e:
                logging.warning(f"Failed to start manual hotkey listener: {e}")
                self.manual_listener = None
                if is_MACOS:
                    self._check_and_warn_macos_permissions('input_monitoring')
                    self._check_and_warn_macos_permissions('accessibility')

        logging.info(
            f"Hotkey listeners started for combos: '{self.hotkey_str}' and '{self.post_rephrase_hotkey_str}'"
        )

    def _on_hotkey_press(self, key: Any, injected: bool = False) -> None:
        """Pynput callback (listener thread) for any key press."""
        key_tokens = self._key_to_hotkey_tokens(key)
        if injected and not self._should_process_injected_hotkey_event(key_tokens):
            return
        self.pressed_hotkey_tokens.update(key_tokens)

        # Snapshot the binding list: the main thread swaps in a new list on re-init.
        for binding in list(self.manual_hotkey_bindings):
            if (hotkeys.binding_matches_current_press(binding, self.pressed_hotkey_tokens, key_tokens)
                    and binding["action"] not in self.active_hotkey_actions):
                self.active_hotkey_actions.add(binding["action"])
                logging.info(f"Manual hotkey combo detected: {binding['display']}")
                if binding["action"] == "transcription" and self.config.get("push_to_talk", False):
                    if not self.is_recording:
                        self.push_to_talk_active = True
                        self.hotkey_action_signal.emit(binding["action"])
                else:
                    self.hotkey_action_signal.emit(binding["action"])
                if binding["action"] == "transcription":
                    return

    def _on_hotkey_release(self, key: Any, injected: bool = False) -> None:
        """Pynput callback (listener thread) for any key release."""
        released_tokens = self._key_to_hotkey_tokens(key)
        if injected and not self._should_process_injected_hotkey_event(released_tokens):
            return
        transcription_binding = self._get_binding_for_action("transcription")
        if (
            self.config.get("push_to_talk", False)
            and self.push_to_talk_active
            and self.is_recording
            and transcription_binding
            and released_tokens.intersection(hotkeys.binding_release_tokens(transcription_binding))
        ):
            logging.info("Push-to-talk hotkey released. Stopping recording.")
            self.push_to_talk_active = False
            self.hotkey_action_signal.emit("stop_transcription")

        self.pressed_hotkey_tokens.difference_update(released_tokens)

        still_active = {
            binding["action"]
            for binding in list(self.manual_hotkey_bindings)
            if hotkeys.binding_matches_pressed(binding, self.pressed_hotkey_tokens)
        }
        self.active_hotkey_actions.intersection_update(still_active)

    # --- "Set hotkey" capture flow -------------------------------------------------------
    def start_hotkey_capture(self) -> None:
        """Initiates the process of listening for a new hotkey."""
        self._check_and_warn_macos_permissions('input_monitoring')

        sender = self.sender()
        if sender == self.set_hotkey_button:
            target_widget = self.hotkey_display
            button_widget = self.set_hotkey_button
        elif sender == self.set_pr_hotkey_button:
            target_widget = self.pr_hotkey_display
            button_widget = self.set_pr_hotkey_button
        else:
            return

        self.capturing_for_widget = target_widget
        self.capturing_button = button_widget
        button_widget.setText(self.translator.tr("hotkey_listening_button"))
        button_widget.setEnabled(False)
        self.captured_keys = set()

        # Suspend the GLOBAL hotkeys while capturing. Otherwise pressing e.g. F9 both records the
        # key AND fires its global action (post-rephrase), whose simulated Ctrl+C then gets caught
        # by the capture listener too — producing garbage like "<ctrl>++<f9>+c".
        self._stop_hotkey_listeners()

        if is_MACOS:
            target_widget.clear()
            target_widget.installEventFilter(self)
            target_widget.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
            target_widget.grabKeyboard()
            return

        self.hotkey_capture_listener = keyboard.Listener(on_press=self.on_press_capture,
                                                         on_release=self.on_release_capture)
        self.hotkey_capture_listener.start()

    def on_press_capture(self, key: Any) -> None:
        """Pynput capture callback (listener thread): record the key, preview via signal.

        Runs on pynput's thread — the widget text is updated through the queued
        ``hotkey_capture_text_signal`` so no Qt object is touched off the main thread.
        """
        if not self.capturing_for_widget:
            return
        self.captured_keys.update(self._key_to_hotkey_tokens(key))
        self.hotkey_capture_text_signal.emit(hotkeys.format_hotkey_tokens(self.captured_keys))

    def on_release_capture(self, key: Any) -> None:
        """Pynput capture callback (listener thread): finalize the capture via signals."""
        if self.hotkey_capture_listener:
            self.hotkey_capture_listener.stop()
            self.hotkey_capture_listener = None

        if self.capturing_for_widget:
            self.hotkey_capture_text_signal.emit(hotkeys.format_hotkey_tokens(self.captured_keys))
            # _finish_hotkey_capture must run on the main thread (it touches buttons and
            # restarts the global listeners) — hand it over via the queued signal.
            self.hotkey_capture_finished_signal.emit()

    def _apply_captured_hotkey_text(self, text: str) -> None:
        """Main-thread slot: show the current capture preview in the target field."""
        if self.capturing_for_widget:
            self.capturing_for_widget.setText(text)

    def _finish_hotkey_capture(self) -> None:
        """Reset the UI after a hotkey capture has completed (main thread)."""
        if self.capturing_for_widget is None and self.capturing_button is None:
            return  # capture already finished/cancelled; a late signal must not re-run this

        if is_MACOS and self.capturing_for_widget:
            self.capturing_for_widget.releaseKeyboard()
            self.capturing_for_widget.removeEventFilter(self)

        if self.capturing_button:
            self.capturing_button.setText(self.translator.tr("set_hotkey_button"))
            self.capturing_button.setEnabled(True)

        self.capturing_for_widget = None
        self.capturing_button = None

        # Restore the global hotkey listeners now that capture is finished.
        self.init_manual_hotkey_listener()

    def _cancel_hotkey_capture(self) -> None:
        """Abort an in-progress hotkey capture and restore the global listeners.

        start_hotkey_capture() stops every global listener so the keys pressed for capture are
        not also fired as hotkeys. If the capture is abandoned (e.g. the settings window is
        closed before a key is pressed), nothing would otherwise restart them and all hotkeys
        stay dead until the next save/restart. This makes that path safe.
        """
        if not self.capturing_for_widget and not self.capturing_button:
            return
        if self.hotkey_capture_listener:
            try:
                self.hotkey_capture_listener.stop()
            except Exception:
                pass
            self.hotkey_capture_listener = None
        logging.info("Hotkey capture aborted; restoring global hotkey listeners.")
        self._finish_hotkey_capture()

    def _on_windows_hotkey_registration_failed(self, display: str) -> None:
        """Surface a Windows hotkey that could not be registered (runs in the listener thread).

        Registration ultimately fails when another application owns the combo. show_tray_balloon
        is thread-safe (it emits a queued signal), so the notice reaches the GUI thread safely.
        """
        logging.warning(f"Windows hotkey {display} could not be registered; it will not fire.")
        try:
            self.show_tray_balloon(self.translator.tr("hotkey_register_failed_message", hotkey=display), 4000)
        except Exception:
            pass

    def _update_windows_keep_mic_hot_ui_state(self) -> None:
        """Enable or disable Windows prewarm idle controls based on the checkbox state."""
        enabled = is_WINDOWS and self.windows_keep_mic_hot_checkbox.isChecked()
        self.windows_keep_mic_hot_idle_label.setEnabled(enabled)
        self.windows_keep_mic_hot_idle_input.setEnabled(enabled)

    # --- Event -> canonical-token conversion (needs Qt / pynput objects) -----------------
    def _qt_key_to_hotkey_tokens(self, event: QKeyEvent) -> Set[str]:
        """Convert a Qt key event to canonical hotkey tokens for macOS capture."""
        tokens: Set[str] = set()
        modifiers = event.modifiers()

        if modifiers & Qt.KeyboardModifier.ControlModifier:
            tokens.add("<ctrl>")
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            tokens.add("<shift>")
        if modifiers & Qt.KeyboardModifier.AltModifier:
            tokens.add("<alt>")
        if modifiers & Qt.KeyboardModifier.MetaModifier:
            tokens.add("<cmd>" if is_MACOS else "<win>")

        key = event.key()
        modifier_keys = {
            Qt.Key.Key_Control,
            Qt.Key.Key_Shift,
            Qt.Key.Key_Alt,
            Qt.Key.Key_Meta,
        }
        if key in modifier_keys:
            return tokens

        qt_special_keys = {
            Qt.Key.Key_Escape: "<esc>",
            Qt.Key.Key_Tab: "<tab>",
            Qt.Key.Key_Backtab: "<tab>",
            Qt.Key.Key_Backspace: "<backspace>",
            Qt.Key.Key_Return: "<enter>",
            Qt.Key.Key_Enter: "<enter>",
            Qt.Key.Key_Insert: "<insert>",
            Qt.Key.Key_Delete: "<delete>",
            Qt.Key.Key_Home: "<home>",
            Qt.Key.Key_End: "<end>",
            Qt.Key.Key_Left: "<left>",
            Qt.Key.Key_Up: "<up>",
            Qt.Key.Key_Right: "<right>",
            Qt.Key.Key_Down: "<down>",
            Qt.Key.Key_PageUp: "<page_up>",
            Qt.Key.Key_PageDown: "<page_down>",
            Qt.Key.Key_CapsLock: "<caps_lock>",
            Qt.Key.Key_Space: "<space>",
        }

        if Qt.Key.Key_F1 <= key <= Qt.Key.Key_F35:
            tokens.add(f"<f{int(key) - int(Qt.Key.Key_F1) + 1}>")
            return tokens

        if is_MACOS:
            macos_function_aliases = {
                getattr(Qt.Key, 'Key_MediaPrevious', None): "<f7>",
                getattr(Qt.Key, 'Key_MediaTogglePlayPause', None): "<f8>",
                getattr(Qt.Key, 'Key_MediaPlay', None): "<f8>",
                getattr(Qt.Key, 'Key_MediaNext', None): "<f9>",
                getattr(Qt.Key, 'Key_VolumeMute', None): "<f10>",
                getattr(Qt.Key, 'Key_VolumeDown', None): "<f11>",
                getattr(Qt.Key, 'Key_VolumeUp', None): "<f12>",
            }
            mapped_key = macos_function_aliases.get(key)
            if mapped_key:
                tokens.add(mapped_key)
                return tokens

        mapped_special_key = qt_special_keys.get(key)
        if mapped_special_key:
            tokens.add(mapped_special_key)
            return tokens

        text = event.text().lower()
        if len(text) == 1 and text.isprintable() and not text.isspace():
            tokens.add(text)
            return tokens

        if Qt.Key.Key_A <= key <= Qt.Key.Key_Z:
            tokens.add(chr(ord("a") + int(key) - int(Qt.Key.Key_A)))
            return tokens

        if Qt.Key.Key_0 <= key <= Qt.Key.Key_9:
            tokens.add(chr(ord("0") + int(key) - int(Qt.Key.Key_0)))
            return tokens

        fallback_key_map = {
            Qt.Key.Key_Minus: "-",
            Qt.Key.Key_Equal: "=",
            Qt.Key.Key_BracketLeft: "[",
            Qt.Key.Key_BracketRight: "]",
            Qt.Key.Key_Backslash: "\\",
            Qt.Key.Key_Semicolon: ";",
            Qt.Key.Key_Apostrophe: "'",
            Qt.Key.Key_Comma: ",",
            Qt.Key.Key_Period: ".",
            Qt.Key.Key_Slash: "/",
            Qt.Key.Key_QuoteLeft: "`",
        }
        fallback_char = fallback_key_map.get(key)
        if fallback_char:
            tokens.add(fallback_char)

        return tokens

    def _should_process_injected_hotkey_event(self, key_tokens: Set[str]) -> bool:
        """Allow macOS remappers to trigger hotkeys when they inject special keys like function keys."""
        return is_MACOS and any(hotkeys.is_non_modifier_special_token(token) for token in key_tokens)

    def _key_to_hotkey_tokens(self, key: Any) -> Set[str]:
        """Convert a pynput key event to canonical hotkey tokens."""
        tokens: Set[str] = set()
        key_name = getattr(key, "name", None)

        special_name_map = {
            "ctrl": "<ctrl>", "ctrl_l": "<ctrl>", "ctrl_r": "<ctrl>",
            "alt": "<alt>", "alt_l": "<alt>", "alt_r": "<alt>", "alt_gr": "<alt_gr>",
            "shift": "<shift>", "shift_l": "<shift>", "shift_r": "<shift>",
            "cmd": "<cmd>", "cmd_l": "<cmd>", "cmd_r": "<cmd>",
            "caps_lock": "<caps_lock>", "esc": "<esc>", "space": "<space>",
            "enter": "<enter>", "tab": "<tab>", "backspace": "<backspace>",
            "delete": "<delete>", "insert": "<insert>", "home": "<home>", "end": "<end>",
            "page_up": "<page_up>", "page_down": "<page_down>",
            "left": "<left>", "right": "<right>", "up": "<up>", "down": "<down>",
            # macOS surfaces F7-F12 as media keys when the Fn toggle is set to media mode.
            "media_previous": "<f7>", "media_play_pause": "<f8>", "media_next": "<f9>",
            "media_volume_mute": "<f10>", "media_volume_down": "<f11>", "media_volume_up": "<f12>",
        }

        if key_name in special_name_map:
            tokens.add(special_name_map[key_name])

        function_key_token = hotkeys.normalize_hotkey_part(key_name or "")
        if function_key_token.startswith("<f") and function_key_token.endswith(">"):
            tokens.add(function_key_token)

        # On Windows, AltGr is often surfaced as the right Alt key plus an implicit Ctrl.
        if is_WINDOWS and key_name == "alt_r":
            tokens.add("<alt_gr>")
            tokens.add("<alt>")
            tokens.add("<ctrl>")

        char = getattr(key, "char", None)
        if char:
            tokens.add(char.lower())

        vk_token = hotkeys.vk_to_key_token(getattr(key, "vk", None))
        if vk_token:
            tokens.add(vk_token)

        if "<alt_gr>" in tokens:
            tokens.add("<ctrl>")
            tokens.add("<alt>")

        return {hotkeys.normalize_hotkey_part(token) for token in tokens}

    def _get_binding_for_action(self, action: str) -> Optional[Dict[str, Any]]:
        """Return the configured binding for a given action, if any."""
        for binding in self.hotkey_bindings:
            if binding["action"] == action:
                return binding
        return None

    def _scroll_macos_area_from_slider_wheel(self, watched: QObject, event: QWheelEvent) -> bool:
        """Route wheel gestures from sliders to the surrounding scroll area on macOS."""
        scroll_area = None
        if watched == self.transcription_temp_slider:
            scroll_area = self.transcription_scroll_area
        elif watched == self.rephrasing_temp_slider:
            scroll_area = self.rephrasing_scroll_area

        if scroll_area is None:
            return False

        scrollbar = scroll_area.verticalScrollBar()
        pixel_delta = event.pixelDelta().y()
        angle_delta = event.angleDelta().y()

        if pixel_delta:
            delta = -pixel_delta
        elif angle_delta:
            steps = angle_delta / 120.0
            delta = int(-steps * max(scrollbar.singleStep(), 18) * 3)
        else:
            return True

        scrollbar.setValue(scrollbar.value() + delta)
        return True
