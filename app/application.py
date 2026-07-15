"""WhisperTyper application class (composition root).

``WhisperTyperApp`` is a deliberately thin coordinator: it owns the shared runtime state
and the cross-thread signals, wires everything together in ``__init__``, and inherits its
behaviour from the domain mixins in ``app/mixins/*`` (see docs/ARCHITECTURE.md for the
layering and threading model). The process entry point + bootstrap (single-instance lock,
stderr redirect, venv re-exec, base logging) live in ``run.py`` / ``app/bootstrap.py``.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Set

import pyaudio
from pynput import keyboard
from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QIcon
from PyQt6.QtWidgets import QApplication, QLineEdit, QMessageBox, QPushButton, QWidget

from app.audio.sound import SoundPlayer
from app.audio.store import RecordingStore
from app.core.env import is_MACOS
from app.core.i18n import TranslationManager
from app.core.paths import resource_path
from app.hotkeys.windows_listener import WindowsHotkeyListener
from app.mixins.audio_mixin import AudioMixin
from app.mixins.clipboard_mixin import ClipboardMixin
from app.mixins.hotkey_mixin import HotkeyMixin
from app.mixins.mac_mixin import MacMixin
from app.mixins.post_rephrase_mixin import PostRephraseMixin
from app.mixins.settings_mixin import SettingsMixin
from app.mixins.theme_mixin import ThemeMixin
from app.mixins.transcription_mixin import TranscriptionMixin
from app.mixins.tray_mixin import TrayMixin
from app.mixins.widget_attrs import WidgetAttrs
from app.services.rephrasing_worker import RephrasingWorker
from app.services.transcription_worker import TranscriptionWorker
from app.ui.floating_buttons import FloatingButtonWindow
from app.ui.tooltip import MouseFollowerTooltip

# Suppress verbose DEBUG messages from the pyuic module
logging.getLogger('PyQt6.uic').setLevel(logging.WARNING)


class WhisperTyperApp(WidgetAttrs, ThemeMixin, MacMixin, TrayMixin, AudioMixin, HotkeyMixin,
                      SettingsMixin, TranscriptionMixin, ClipboardMixin, PostRephraseMixin, QWidget):
    """
    Main application class for WhisperTyper.
    Manages the GUI, system tray icon, audio recording, and hotkey listeners.
    """

    # Queued signals for thread-safe GUI updates: worker/listener threads emit these and the
    # connected slots run on the Qt main thread.
    show_tooltip_signal = pyqtSignal(str, int, bool, bool)
    hide_tooltip_signal = pyqtSignal()
    show_floating_window_signal = pyqtSignal(list, str)
    show_permission_dialog_signal = pyqtSignal(str, str, str)
    hotkey_action_signal = pyqtSignal(str)
    # Hotkey capture runs on a pynput listener thread; these marshal its UI updates
    # (field preview text / capture teardown) onto the main thread.
    hotkey_capture_text_signal = pyqtSignal(str)
    hotkey_capture_finished_signal = pyqtSignal()

    def __init__(self) -> None:
        """Initializes the application."""
        super().__init__()
        self.keyboard_controller = keyboard.Controller()
        self.is_recording: bool = False
        self.recorded_frames: List[bytes] = []
        self.samplerate: int = 16000
        self.chunk_size: int = 1024
        self.input_pyaudio_instance: Optional[pyaudio.PyAudio] = None

        self.config: Dict[str, Any] = {}
        self.load_config()  # Load config first to get UI language

        # Initialize TranslationManager
        self.translator = TranslationManager(initial_language=self.config.get('ui_language', 'en'))

        # Add placeholder for file log handler
        self._file_log_handler = None

        # self.hotkey_str is now correctly set within load_config()
        self.capturing_for_widget: Optional[QLineEdit] = None
        self.capturing_button: Optional[QPushButton] = None
        self.captured_keys: Set[Any] = set()

        self.recording_thread: Optional[threading.Thread] = None
        self.hotkey_capture_listener: Optional[keyboard.Listener] = None
        self.manual_listener: Optional[keyboard.Listener] = None
        self.windows_hotkey_listener: Optional[WindowsHotkeyListener] = None
        self.active_workers: List[TranscriptionWorker] = []
        self.active_threads: List[QThread] = []
        self.active_rephrasing_workers: List[RephrasingWorker] = []
        self.active_rephrasing_threads: List[QThread] = []
        self.last_transcription: str = ""
        self.current_transcription_context: str = ""
        self.hotkey_bindings: List[Dict[str, Any]] = []
        self.manual_hotkey_bindings: List[Dict[str, Any]] = []
        self.pressed_hotkey_tokens: Set[str] = set()
        self.active_hotkey_actions: Set[str] = set()
        self.push_to_talk_active = False
        self.audio_state_lock = threading.Lock()
        self.audio_capture_running = False
        self.audio_capture_thread: Optional[threading.Thread] = None
        self.input_stream = None
        self.current_input_samplerate: int = self.samplerate
        self.current_input_device_index: Optional[int] = None
        self.current_input_device_name: str = ""
        self.macos_audio_recorder = None
        self.macos_recording_path: Optional[str] = None
        self.pre_record_buffer = deque(maxlen=max(1, int(self.samplerate * 0.75 / self.chunk_size)))
        self.last_transcription_activity_ts = time.monotonic()
        self.latest_audio_level: float = 0.0
        self._idle_tray_icon: Optional[QIcon] = None
        self._recording_icon_cache: Dict[tuple, QIcon] = {}
        self._recording_tray_timer = QTimer(self)
        self._recording_tray_timer.setInterval(80)
        self._clipboard_restore_timer = QTimer(self)
        self._clipboard_restore_timer.setSingleShot(True)
        self._clipboard_restore_timer.timeout.connect(self._perform_clipboard_restore)
        self._pending_clipboard_restore_state: Optional[Dict[str, Any]] = None
        self._macos_startup_permissions_requested = False
        self._macos_hotkey_permissions_checked = False

        # Single source of truth for on-disk recordings (see app/audio/store.py).
        self.recordings = RecordingStore()
        self.cleanup_old_recordings()

        # After loading config ensure logging handlers reflect settings
        self.apply_logging_configuration()
        self.init_ui()
        self.init_tray_icon()

        self.init_manual_hotkey_listener()
        # Preload short sound effects & prepare reusable output streams for low latency
        self.sound_player = SoundPlayer(resource_path)
        self.sound_player.preload(["sound_start.wav", "sound_end.wav"])

        # Connect the signals to their slots for safe cross-thread communication
        self.show_tooltip_signal.connect(self._show_tooltip_slot)
        self.hide_tooltip_signal.connect(self._hide_tooltip_slot)
        self.show_floating_window_signal.connect(self._show_floating_window_slot)
        self.show_permission_dialog_signal.connect(self._show_permission_dialog_slot)
        self.hotkey_action_signal.connect(self._handle_hotkey_action)
        self.hotkey_capture_text_signal.connect(self._apply_captured_hotkey_text)
        self.hotkey_capture_finished_signal.connect(self._finish_hotkey_capture)
        self._recording_tray_timer.timeout.connect(self._update_recording_tray_icon)

        self.tray_icon.setToolTip(self.translator.tr("tray_ready_tooltip", hotkey=self.hotkey_str))
        logging.info(f"Application started. Press '{self.hotkey_str}' to start/stop recording.")
        self.show_tray_balloon(self.translator.tr("tray_started_message"), 2000)

        if self._use_windows_keep_mic_hot():
            self._start_background_audio_capture()

        # Initial state update for menu actions
        self.update_logfile_menu_action()
        self.update_play_last_recording_action()
        self._update_rephrase_api_group_style()  # Set initial style
        self._update_transcription_api_group_style()  # Set initial style

        if self._should_request_macos_startup_permissions():
            QTimer.singleShot(900, self._request_macos_startup_permissions)

        # On a fresh install no API key is configured yet. Recording is impossible
        # in that state, so open the settings window right away to guide the user.
        if not self._has_valid_api_settings():
            logging.info("No valid API key configured on startup; opening settings window.")
            QTimer.singleShot(600, self.show_settings_window)

    def _show_tooltip_slot(self, message: str, timeout_ms: int, spinner: bool, check: bool) -> None:
        """
        This slot is executed in the main GUI thread and can safely update the UI.

        Args:
            message (str): The message to display in the tooltip.
            timeout_ms (int): The duration in milliseconds for the tooltip to be visible.
            spinner (bool): Whether to show a persistent animated spinner.
            check (bool): Whether to prepend a static green completion checkmark.
        """
        MouseFollowerTooltip.show_tooltip(message, timeout_ms, spinner, check)

    def _hide_tooltip_slot(self) -> None:
        """Main-thread slot that dismisses the current tooltip (ends the spinner state)."""
        MouseFollowerTooltip.hide_tooltip()

    def _show_floating_window_slot(self, valid_entries: List[Dict[str, str]], selected_text: str) -> None:
        """
        This slot is executed in the main GUI thread and can safely create the floating window.

        Args:
            valid_entries (List[Dict[str, str]]): The list of button configurations.
            selected_text (str): The text that was selected.
        """
        FloatingButtonWindow(
            buttons=valid_entries,
            selected_text=selected_text,
            on_button_click_callback=self.on_floating_button_clicked
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Overrides the close event to hide the window instead of quitting.

        Args:
            event (QCloseEvent): The close event.
        """
        # If a hotkey capture was still in progress, abort it so the global listeners are
        # restored — otherwise closing the window mid-capture leaves every hotkey dead.
        self._cancel_hotkey_capture()

        # Remember the window size so it reopens at the same dimensions.
        try:
            self.config["window_width"] = self.width()
            self.config["window_height"] = self.height()
            self.save_config()
        except Exception as e:
            logging.debug(f"Could not persist window size on close: {e}")

        event.ignore()
        self.hide()

        # After closing, delete all old recordings except the last one
        self.recordings.keep_only_latest()

    def quit_app(self) -> None:
        """Quits the application cleanly, optionally after a confirmation dialog."""
        # Skip the confirmation dialog entirely if the user opted in.
        if not self.config.get("quit_without_confirmation", False):
            reply = QMessageBox.question(self, self.translator.tr("quit_dialog_title"),
                                         self.translator.tr("quit_dialog_text"),
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)

            if reply == QMessageBox.StandardButton.No:
                return

        logging.info("Quitting application.")
        self._stop_hotkey_listeners()
        self._stop_background_audio_capture()
        self._drain_worker_threads()
        if is_MACOS and self.macos_audio_recorder:
            self._stop_macos_native_recording(discard=True)
        if self._clipboard_restore_timer.isActive():
            self._clipboard_restore_timer.stop()
            self._perform_clipboard_restore()
        try:
            # Close sound playback streams + PyAudio (owned by SoundPlayer)
            if getattr(self, 'sound_player', None) is not None:
                self.sound_player.close()
            if self.input_pyaudio_instance:
                self.input_pyaudio_instance.terminate()
        except Exception as e:
            logging.debug(f"Audio teardown raised during quit: {e}")
        try:
            if getattr(self, '_file_log_handler', None):
                logging.getLogger().removeHandler(self._file_log_handler)
                try:
                    self._file_log_handler.close()
                except Exception:
                    pass
                self._file_log_handler = None
        except Exception as e:
            logging.debug(f"Log-handler teardown raised during quit: {e}")
        self.tray_icon.hide()
        QApplication.instance().quit()
