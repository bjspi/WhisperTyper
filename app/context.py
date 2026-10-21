"""Shared runtime dependencies handed to every controller and window by the composition root."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Mapping, Optional

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

from app.audio.store import RecordingStore
from app.core.config_store import ConfigStore
from app.core.i18n import TranslationManager
from app.ui import theme
from app.ui.durations import BALLOON_SHORT_MS, BALLOON_WARNING_MS
from app.ui.tooltip import MouseFollowerTooltip


class Notifier(QObject):
    """Mouse-following status balloon, safe to call from any thread.

    The signals are connected to this GUI-thread object, so calls from listener, capture or
    worker threads are queued onto the GUI thread; calls on the GUI thread run directly.
    """

    _show = pyqtSignal(str, int, bool, bool, str, str)
    _hide = pyqtSignal()

    def __init__(self, dark: Optional[Callable[[], bool]] = None) -> None:
        """Connect the cross-thread signals to slots of this GUI-thread object.

        ``dark`` tells, on the GUI thread, whether the balloon is drawn in the dark palette (the
        app's resolved colour theme); without it the balloon follows the operating system.
        """
        super().__init__()
        self._dark = dark
        self._show.connect(self._show_slot)
        self._hide.connect(self._hide_slot)

    def _show_slot(self, message: str, timeout_ms: int, spinner: bool, check: bool, detail: str,
                   detail_level: str) -> None:
        """GUI thread: display the balloon."""
        dark = self._dark() if self._dark is not None else None
        MouseFollowerTooltip.show_tooltip(message, timeout_ms, spinner, check, detail, detail_level, dark)

    def _hide_slot(self) -> None:
        """GUI thread: dismiss the balloon."""
        MouseFollowerTooltip.hide_tooltip()

    def show(self, message: str, timeout_ms: int = BALLOON_SHORT_MS, spinner: bool = False,
             check: bool = False, detail: str = "", detail_level: str = "") -> None:
        """Show ``message``; a spinner stays until hidden or replaced, ``check`` marks completion.

        ``detail`` follows a checkmark message in brackets, coloured by ``detail_level`` ("good"/"warn"/"bad").
        """
        self._show.emit(message, timeout_ms, spinner, check, detail, detail_level)

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
        #: Saved values of keys the open settings form holds unsaved edits of (see ``set_unsaved_values``).
        self._unsaved_values: Optional[Callable[[], Mapping[str, Any]]] = None
        #: Whether the last full save reached the disk.
        self.last_save_ok = True
        if migrated:
            store.save(self.config)
        self.translator = TranslationManager(initial_language=self.config.get("ui_language", "en"))
        self.notifier = Notifier(dark=lambda: theme.resolve_dark(self.config.get("color_theme", "system"),
                                                                  QApplication.instance()))
        self.recordings = recordings

    def tr(self, key: str, **kwargs: Any) -> str:  # type: ignore[override]  # Qt's static tr() is unused
        """Translate ``key`` in the current UI language."""
        return self.translator.tr(key, **kwargs)

    def prune_recordings(self, startup: bool = False) -> None:
        """Delete recordings past ``recording_retention_minutes``; outside startup the newest always stays."""
        try:
            minutes = max(0.0, float(self.config.get("recording_retention_minutes", 0)))
        except (TypeError, ValueError):
            minutes = 0.0
        if startup and minutes == 0:
            self.recordings.cleanup_all()
        else:
            self.recordings.prune(minutes * 60, keep_newest=not startup)

    def set_unsaved_values(self, provider: Optional[Callable[[], Mapping[str, Any]]]) -> None:
        """Register the settings form's unsaved edits: ``provider`` returns their saved values.

        The form mirrors some edits (prompts, UI language) into the live config before its Save.
        A save from elsewhere (microphone picked in the palette, …) writes the saved values of
        those keys instead, so closing the form without saving really discards the edits.
        """
        self._unsaved_values = provider

    def save_config(self, *, with_unsaved_edits: bool = False, notify: bool = True) -> bool:
        """Persist the current configuration; False when writing failed.

        ``with_unsaved_edits`` is the settings form's own Save. A failed write is shown in a
        warning balloon unless ``notify`` is off (the caller reports it itself).
        """
        config = self.config
        if not with_unsaved_edits and self._unsaved_values is not None:
            saved = self._unsaved_values()
            if saved:
                config = {**self.config, **saved}
        ok = bool(self._store.save(config))
        self.last_save_ok = ok
        if not ok and notify:
            self._report_save_failure()
        return ok

    def save_error(self) -> str:
        """Why the last save failed."""
        return str(getattr(self._store, "last_error", "") or "?")

    def _report_save_failure(self) -> None:
        self.notifier.show(self.tr("config_save_failed_message", error=self.save_error()), BALLOON_WARNING_MS)

    def save_config_keys(self, *keys: str) -> None:
        """Persist only ``keys`` of the current configuration; the file keeps its other values.

        For values that apply at once (window size, colour theme) while the settings form may
        still hold unsaved edits mirrored into the live config. An unreadable file (e.g. a typo
        from editing it by hand) is left as it is, instead of being replaced by the defaults.
        """
        if self._store.read_existing() is None:
            logging.warning("Config file is unreadable; %s not saved and the file left untouched.", ", ".join(keys))
            return
        stored, _migrated = self._store.load()
        for key in keys:
            if key in self.config:
                stored[key] = self.config[key]
        if not self._store.save(stored):
            self._report_save_failure()
