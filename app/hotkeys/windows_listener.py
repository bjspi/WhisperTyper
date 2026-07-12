"""Windows OS-level hotkey registration thread."""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List, Optional

from app.core.env import is_WINDOWS
from app.core.frameworks import MOD_NOREPEAT, WM_HOTKEY, WM_QUIT

if is_WINDOWS:
    import ctypes
    from ctypes import wintypes

class WindowsHotkeyListener(threading.Thread):
    """Registers bindable Windows hotkeys with the OS and dispatches callbacks."""

    # RegisterHotKey transiently fails right after a restart while the previous instance still
    # owns the hotkey (or the OS hasn't reclaimed it yet). Retry for a few seconds so it recovers
    # on its own instead of leaving the hotkey silently dead until the next restart.
    _REGISTER_RETRIES = 15
    _REGISTER_RETRY_INTERVAL = 0.2

    def __init__(self, bindings: List[Dict[str, Any]], callback: Callable[[str], None],
                 on_registration_failed: Optional[Callable[[str], None]] = None) -> None:
        """Store the hotkey bindings, the action callback and an optional failure notifier."""
        super().__init__(daemon=True)
        self.bindings = {binding["id"]: binding for binding in bindings}
        self.callback = callback
        self.on_registration_failed = on_registration_failed
        self._thread_id: Optional[int] = None
        self._stop_event = threading.Event()

    def _register_with_retry(self, user32: Any, hotkey_id: int, modifiers: int, vk: int, display: str) -> bool:
        """Try to register one hotkey, retrying briefly to ride out a post-restart ownership race."""
        for attempt in range(self._REGISTER_RETRIES):
            if self._stop_event.is_set():
                return False
            if user32.RegisterHotKey(None, hotkey_id, modifiers, vk):
                if attempt:
                    logging.info(f"Registered Windows hotkey {display} after {attempt + 1} attempts.")
                else:
                    logging.info(f"Registered Windows hotkey: {display}")
                return True
            # Wait (interruptibly) before the next attempt.
            if self._stop_event.wait(self._REGISTER_RETRY_INTERVAL):
                return False
        return False

    def run(self) -> None:
        """Register the hotkeys and pump the Windows message loop until stopped."""
        if not is_WINDOWS:
            return

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        self._thread_id = kernel32.GetCurrentThreadId()
        registered_ids: List[int] = []

        try:
            for hotkey_id, binding in self.bindings.items():
                modifiers = binding["windows_modifiers"] | MOD_NOREPEAT
                if self._register_with_retry(user32, hotkey_id, modifiers, binding["windows_vk"], binding["display"]):
                    registered_ids.append(hotkey_id)
                else:
                    logging.warning(
                        f"Failed to register Windows hotkey after {self._REGISTER_RETRIES} attempts: {binding['display']}"
                    )
                    if self.on_registration_failed and not self._stop_event.is_set():
                        try:
                            self.on_registration_failed(binding["display"])
                        except Exception as e:
                            logging.debug(f"Hotkey registration-failure notifier raised: {e}")

            msg = wintypes.MSG()
            while not self._stop_event.is_set():
                result = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if result <= 0:
                    break
                if msg.message == WM_HOTKEY:
                    fired = self.bindings.get(int(msg.wParam))
                    if fired:
                        self.callback(fired["action"])
        finally:
            for hotkey_id in registered_ids:
                try:
                    user32.UnregisterHotKey(None, hotkey_id)
                except Exception:
                    pass

    def stop(self) -> None:
        """Signal the message loop to exit and wake the thread."""
        self._stop_event.set()
        if is_WINDOWS and self._thread_id:
            try:
                ctypes.windll.user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
            except Exception:
                pass
