"""WhisperTyper composition root.

``WhisperTyperApp`` builds the shared context, the controllers and the windows, wires their
signals and owns the startup/quit lifecycle. Every component receives its collaborators in
its constructor (see docs/ARCHITECTURE.md for the layering and threading model). The process
entry point and bootstrap (single-instance lock, stderr redirect, venv re-exec, base logging)
live in ``run.py`` / ``app/bootstrap.py``.
"""
from __future__ import annotations

import logging
import os
import sys
from logging import Handler
from typing import Optional

from PyQt6.QtCore import QObject, QProcess, QTimer
from PyQt6.QtWidgets import QApplication, QMessageBox

from app.audio.sound import SoundPlayer
from app.audio.store import RecordingStore
from app.context import AppContext
from app.controllers.file_actions import FileActions
from app.controllers.hotkeys import HotkeyController
from app.controllers.permissions import MacPermissions
from app.controllers.post_rephrase import PostRephraseController
from app.controllers.recording import RecordingController
from app.controllers.text_output import TextOutput
from app.controllers.transcription import TranscriptionPipeline
from app.controllers.tray import TrayController
from app.controllers.warmup import WarmupScheduler
from app.controllers.workers import WorkerThreads
from app.core.api_keys import transcription_configured
from app.core.config_store import ConfigStore
from app.core.constants import CONFIG_FILE
from app.core.hotkeys import normalize_hotkey_string
from app.core.paths import resource_path
from app.services.http_transport import close_transport
from app.services.logging_config import apply_logging_config
from app.ui.durations import BALLOON_INFO_MS, BALLOON_SHORT_MS
from app.ui.settings.window import SettingsWindow

# Suppress verbose DEBUG messages from the pyuic module
logging.getLogger('PyQt6.uic').setLevel(logging.WARNING)


class WhisperTyperApp(QObject):
    """Builds and connects the application's components; owns startup and shutdown."""

    def __init__(self) -> None:
        """Create every component, wire them together and start the listeners."""
        super().__init__()
        self.ctx = AppContext(ConfigStore(CONFIG_FILE, normalize_hotkey_string), RecordingStore())
        config = self.ctx.config
        self.ctx.recordings.cleanup_all()
        self._file_log_handler: Optional[Handler] = None
        self.apply_logging_configuration()

        self.workers = WorkerThreads()
        self.warmup = WarmupScheduler(config, is_recording=lambda: self.recording.is_recording,
                                      rephrasing_first=lambda: self.recording.rephrasing_expected())
        self.permissions = MacPermissions(self.ctx, dialog_parent=lambda: self.settings)
        self.text_output = TextOutput(config, self.ctx.tr, self.ctx.notifier.show, self.permissions.warn)
        self.pipeline = TranscriptionPipeline(self.ctx, self.text_output, self.warmup, self.workers)
        # Preload short sound effects & prepare reusable output streams for low latency.
        self.sound_player = SoundPlayer(resource_path)
        self.sound_player.preload(["sound_start.wav", "sound_end.wav"])
        self.recording = RecordingController(
            self.ctx, permissions=self.permissions, warmup=self.warmup, output=self.text_output,
            pipeline=self.pipeline, sounds=self.sound_player,
            open_settings=lambda: self.settings.show_window(), palette_anchor=lambda: self.tray.anchor(),
        )
        self.hotkeys = HotkeyController(self.ctx, self.recording, self.permissions)
        self.post_rephrase = PostRephraseController(self.ctx, self.text_output, self.pipeline)
        self.files = FileActions(self.ctx)
        self.settings = SettingsWindow(self.ctx, recording=self.recording, hotkeys=self.hotkeys,
                                       files=self.files, quit_app=self.quit_app)
        self.tray = TrayController(self.ctx, self.settings, self.settings, recording=self.recording,
                                   pipeline=self.pipeline, files=self.files,
                                   restart=self.restart_app, quit_app=self.quit_app)
        self.tray.build()
        self.hotkeys.restart()

        # Hotkey actions arrive from listener threads (queued onto the GUI thread).
        self.hotkeys.action_triggered.connect(self._on_hotkey_action)
        self.settings.saved.connect(self._on_settings_saved)
        self.settings.hotkeys_changed.connect(self.hotkeys.restart)
        self.settings.language_changed.connect(self._on_language_changed)

        self.tray.set_tooltip(self.ctx.tr("tray_ready_tooltip", hotkey=config["hotkey"]))
        logging.info(f"Application started. Press '{config['hotkey']}' to start/stop recording.")
        self.ctx.notifier.show(self.ctx.tr("tray_started_message"), BALLOON_SHORT_MS)

        if self.recording.keep_mic_hot():
            self.recording.start_background_capture()

        QTimer.singleShot(0, self.settings.start_aac_bitrate_probe)

        if self.permissions.startup_prompts_wanted():
            QTimer.singleShot(900, lambda: self.permissions.request_at_startup(self.recording.probe_microphone))

        # On a fresh install no API key is configured yet. Recording is impossible
        # in that state, so open the settings window right away to guide the user.
        if not transcription_configured(config):
            logging.info("No valid API key configured on startup; opening settings window.")
            QTimer.singleShot(600, self.settings.show_window)

    def _on_hotkey_action(self, action: str, detected_ns: Optional[int] = None) -> None:
        """Dispatch a hotkey action (GUI thread)."""
        recording = self.recording
        if action == "transcription":
            if self.ctx.config.get("push_to_talk", False):
                if not recording.is_recording:
                    recording.push_to_talk_active = True
                    recording.toggle(detected_ns=detected_ns)
            else:
                recording.toggle(detected_ns=detected_ns)
        elif action == "stop_transcription":
            if recording.is_recording:
                recording.push_to_talk_active = False
                recording.toggle(detected_ns=detected_ns)
        elif action == "post_rephrase":
            self.post_rephrase.trigger()

    def _on_settings_saved(self) -> None:
        """Apply a saved form to the running components."""
        self.recording.apply_settings()
        self.pipeline.reload_replacements()
        self.ctx.notifier.show(self.ctx.tr("settings_saved_message", hotkey=self.ctx.config["hotkey"]), BALLOON_SHORT_MS)
        self.apply_logging_configuration()
        self.warmup.schedule(activate=True)
        self.ctx.files_changed.emit()

    def _on_language_changed(self) -> None:
        """Rebuild the tray menu in the new language."""
        self.tray.build()

    def apply_logging_configuration(self) -> None:
        """Apply the saved logging level, redaction and file logging."""
        self._file_log_handler = apply_logging_config(self.ctx.config, self._file_log_handler)

    def restart_app(self) -> None:
        """Relaunch WhisperTyper: spawn a fresh instance (``-r`` reclaims the lock), then exit.

        The global listeners and background capture stop first, so the low-level keyboard hook
        is gone before the new instance's ``-r`` hard-stops this one; an orphaned hook would
        otherwise cause system-wide input lag.
        """
        if getattr(sys, "frozen", False):
            program, arguments = sys.executable, ["-r"]
        else:
            program, arguments = sys.executable, [os.path.abspath(sys.argv[0]), "-r"]

        logging.info(f"Restarting application: {program} {arguments}")
        try:
            self.hotkeys.stop()
            self.recording.stop_background_capture()
        except Exception:
            pass

        if not QProcess.startDetached(program, arguments):
            logging.error("Failed to launch a new instance for restart.")
            self.ctx.notifier.show(self.ctx.tr("restart_failed_message"), BALLOON_INFO_MS)
            return
        app = QApplication.instance()
        if app is not None:
            app.quit()

    def quit_app(self) -> None:
        """Quit cleanly, optionally after a confirmation dialog."""
        tr = self.ctx.tr
        if not self.ctx.config.get("quit_without_confirmation", False):
            reply = QMessageBox.question(self.settings, tr("quit_dialog_title"), tr("quit_dialog_text"),
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return

        logging.info("Quitting application.")
        self.warmup.close()
        self.recording.abandon_prompt_selection()
        self.hotkeys.stop()
        self.recording.shutdown()
        self.workers.drain()
        close_transport()
        self.text_output.restore_clipboard_now()
        try:
            # Close sound playback streams + PyAudio (owned by SoundPlayer)
            self.sound_player.close()
        except Exception as e:
            logging.debug(f"Audio teardown raised during quit: {e}")
        try:
            if self._file_log_handler is not None:
                logging.getLogger().removeHandler(self._file_log_handler)
                try:
                    self._file_log_handler.close()
                except Exception:
                    pass
                self._file_log_handler = None
        except Exception as e:
            logging.debug(f"Log-handler teardown raised during quit: {e}")
        self.tray.hide()
        app = QApplication.instance()
        if app is not None:
            app.quit()
