"""Shared runtime dependencies handed to every controller and window by the composition root."""
from __future__ import annotations

from typing import Any, Dict

from PyQt6.QtCore import QObject, pyqtSignal

from app.audio.store import RecordingStore
from app.core.config_store import ConfigStore
from app.core.i18n import TranslationManager
from app.ui.durations import BALLOON_SHORT_MS
from app.ui.tooltip import MouseFollowerTooltip


class Notifier(QObject):
    """Mouse-following status balloon, safe to call from any thread.

    The signals are connected to this GUI-thread object, so calls from listener, capture or
    worker threads are queued onto the GUI thread; calls on the GUI thread run directly.
    """

    _show = pyqtSignal(str, int, bool, bool)
    _hide = pyqtSignal()

    def __init__(self) -> None:
        """Connect the cross-thread signals to slots of this GUI-thread object."""
        super().__init__()
        self._show.connect(self._show_slot)
        self._hide.connect(self._hide_slot)

    def _show_slot(self, message: str, timeout_ms: int, spinner: bool, check: bool) -> None:
        """GUI thread: display the balloon."""
        MouseFollowerTooltip.show_tooltip(message, timeout_ms, spinner, check)

    def _hide_slot(self) -> None:
        """GUI thread: dismiss the balloon."""
        MouseFollowerTooltip.hide_tooltip()

    def show(self, message: str, timeout_ms: int = BALLOON_SHORT_MS, spinner: bool = False,
             check: bool = False) -> None:
        """Show ``message``; a spinner stays until hidden or replaced, ``check`` marks completion."""
        self._show.emit(message, timeout_ms, spinner, check)

    def hide(self) -> None:
        """Dismiss the current balloon (ends a persistent spinner)."""
        self._hide.emit()


class AppContext(QObject):
    """Configuration, translations, notifications and recordings shared by all components."""

    #: Recording or log-file availability may have changed; menus refresh their actions.
    files_changed = pyqtSignal()

    def __init__(self, store: ConfigStore, recordings: RecordingStore) -> None:
        """Load the configuration (with migrations) and create the translator for its UI language."""
        super().__init__()
        self._store = store
        self.config: Dict[str, Any]
        self.config, migrated = store.load()
        if migrated:
            self.save_config()
        self.translator = TranslationManager(initial_language=self.config.get("ui_language", "en"))
        self.notifier = Notifier()
        self.recordings = recordings

    def tr(self, key: str, **kwargs: Any) -> str:  # type: ignore[override]  # Qt's static tr() is unused
        """Translate ``key`` in the current UI language."""
        return self.translator.tr(key, **kwargs)

    def save_config(self) -> None:
        """Persist the current configuration."""
        self._store.save(self.config)
