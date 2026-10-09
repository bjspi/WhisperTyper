"""TrayMixin — system-tray icon, context menu and recording-state icon."""
from __future__ import annotations

import logging
import os
import sys
import time
from typing import Optional

from PyQt6.QtCore import QProcess, QTimer
from PyQt6.QtGui import QAction, QCursor, QIcon
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox, QStyle, QSystemTrayIcon

from app.core.api_keys import transcription_configured
from app.core.env import is_MACOS
from app.services.ffmpeg import VIDEO_EXTENSIONS, is_video_file, resolve_ffmpeg
from app.services.gitutil import git_available
from app.services.updater import GitUpdater
from app.ui import macos_icons, theme
from app.ui.durations import BALLOON_INFO_MS, BALLOON_SHORT_MS, BALLOON_WARNING_MS
from app.ui.tray_icons import app_icon, recording_level_icon
from app.ui.tray_menu import BadgeTrayMenu


class TrayMixin:
    """System-tray icon, context menu and recording-state icon."""

    def _build_recording_tray_icon(self, level: float) -> QIcon:
        """Tray icon for the recording state, reflecting the live input level."""
        if is_MACOS:
            macos_icon = macos_icons.recording_tray_icon(level, self._get_app_icon())
            if not macos_icon.isNull():
                return macos_icon
        return recording_level_icon(level, self._recording_icon_cache)

    def _set_idle_tray_icon(self) -> None:
        """Restore the non-recording tray icon."""
        if self._recording_tray_timer.isActive():
            self._recording_tray_timer.stop()
        self.latest_audio_level = 0.0
        if self._idle_tray_icon:
            self.tray_icon.setIcon(self._idle_tray_icon)

    def _set_recording_tray_icon_active(self) -> None:
        """Switch the tray icon into animated recording mode."""
        self.latest_audio_level = 0.0
        self._update_recording_tray_icon()
        if not self._recording_tray_timer.isActive():
            self._recording_tray_timer.start()

    def _update_recording_tray_icon(self) -> None:
        """Refresh the tray icon with the latest measured input level."""
        if not self.is_recording:
            self._set_idle_tray_icon()
            return
        native_level = self._mac_recorder.level()
        if native_level is not None:
            self.latest_audio_level = native_level
        self.tray_icon.setIcon(self._build_recording_tray_icon(self.latest_audio_level))

    def show_tray_balloon(self, message: str, timeout_ms: int = BALLOON_SHORT_MS, spinner: bool = False,
                          check: bool = False) -> None:
        """
        Shows a custom tooltip by emitting a signal to the main thread.
        This is the thread-safe way to show tooltips from any thread.

        Args:
            message (str): The message to display.
            timeout_ms (int): The duration in milliseconds.
            spinner (bool): If True, show a persistent animated spinner that stays until it is
                explicitly hidden (hide_tray_balloon) or replaced by another balloon.
            check (bool): If True, prepend a static green completion checkmark (replaces the
                spinner when a job finishes). Ignored if ``spinner`` is True.
        """
        self.show_tooltip_signal.emit(message, timeout_ms, spinner, check)

    def hide_tray_balloon(self) -> None:
        """Dismiss the current tooltip (thread-safe). Used to end a persistent spinner balloon."""
        self.hide_tooltip_signal.emit()

    def init_tray_icon(self) -> None:
        """Initializes the system tray icon and its context menu."""
        # Delay a left-click just long enough to distinguish it from the existing double-click
        # gesture. Without that arbitration, opening the menu on the first release would prevent
        # the second click from reaching the tray icon and break double-click-to-copy.
        if not hasattr(self, "_tray_single_click_timer"):
            self._tray_single_click_timer = QTimer(self)
            self._tray_single_click_timer.setSingleShot(True)
            self._tray_single_click_timer.timeout.connect(self._show_tray_menu)
            self._ignore_tray_trigger_until = 0.0
        else:
            self._tray_single_click_timer.stop()

        # Fully dispose of a previous tray icon + menu before creating new ones.
        # They were parented to `self`, so merely reassigning the Python attribute
        # leaks the old QSystemTrayIcon, QMenu and all its QActions under the Qt parent
        # on every re-init (e.g. each language change). deleteLater() schedules real
        # C++ destruction so re-inits no longer accumulate objects.
        if hasattr(self, 'tray_icon') and self.tray_icon is not None:
            self.tray_icon.hide()
            self.tray_icon.deleteLater()
        if getattr(self, 'tray_menu', None) is not None:
            self.tray_menu.deleteLater()
            self.tray_menu = None

        self.tray_icon = QSystemTrayIcon(self)
        if is_MACOS:
            self._idle_tray_icon = macos_icons.tray_icon()
            if self._idle_tray_icon.isNull():
                self._idle_tray_icon = self._get_app_icon()
            if self._idle_tray_icon.isNull():
                self._idle_tray_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        else:
            self._idle_tray_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        self.tray_icon.setIcon(self._idle_tray_icon)

        # IMPORTANT: The QMenu must have a parent AND a persistent reference on the Python side.
        # QSystemTrayIcon.setContextMenu() does not take ownership, so a bare local QMenu()
        # gets garbage-collected once this method returns — leaving a "ghost" menu whose
        # actions never fire (their triggered signals are dead). Parenting to self + storing
        # self.tray_menu keeps the menu and its actions alive for the app's lifetime.
        tray_menu = BadgeTrayMenu(self, lambda: getattr(self, "_theme_palette", None) or theme.palette(False))
        self.tray_menu = tray_menu

        _sp = QStyle.StandardPixmap

        def _ico(pix: QStyle.StandardPixmap) -> QIcon:
            return self.style().standardIcon(pix)

        def _add(pix: QStyle.StandardPixmap, label_key: str, key: str) -> QAction:
            # BadgeTrayMenu paints <key> as a rounded badge on the far right and triggers the
            # action when <key> is pressed. Letters are fixed in code so they stay unique/stable
            # regardless of the UI language.
            action = tray_menu.addAction(_ico(pix), self.translator.tr(label_key))
            assert action is not None
            tray_menu.register_badge(key, action)
            return action

        # --- Settings / config ---
        show_action = _add(_sp.SP_FileDialogDetailedView, "tray_settings_action", "k")
        show_action.triggered.connect(self.show_settings_window)

        # Add "Open Config" Link
        config_action = _add(_sp.SP_FileIcon, "menu_file_open_config", "o")
        config_action.triggered.connect(self.open_config_file)

        transformations_action = _add(
            _sp.SP_FileDialogListView, "tray_edit_transformations_action", "t"
        )
        transformations_action.triggered.connect(self.show_transformations_settings)

        tray_menu.addSeparator()

        # --- Transcription / recording ---
        copy_action = _add(_sp.SP_FileDialogContentsView, "tray_copy_action", "c")
        copy_action.triggered.connect(self.copy_last_transcription_to_clipboard)

        # Add "Re-transcribe Last Recording" action (result -> clipboard)
        self.retranscribe_action = _add(_sp.SP_BrowserReload, "tray_retranscribe_action", "r")
        self.retranscribe_action.triggered.connect(self.retranscribe_last_recording)
        self.retranscribe_action.setEnabled(False)  # Disabled until a recording exists

        # Add "Transcribe Audio File..." action (result -> clipboard)
        transcribe_file_action = _add(_sp.SP_DialogOpenButton, "tray_transcribe_file_action", "f")
        transcribe_file_action.triggered.connect(self.transcribe_audio_file)

        # Add "Play Last Recording" action
        self.play_action = _add(_sp.SP_MediaPlay, "tray_play_action", "p")
        self.play_action.triggered.connect(self.play_latest_recording)
        self.play_action.setEnabled(False)  # Disabled until a recording exists

        # "Cancel Recording" — discards the in-progress recording (no transcription, no clipboard
        # change, and it does NOT touch the stored "last recording"). Enabled only while recording.
        self.cancel_action = _add(_sp.SP_DialogCancelButton, "tray_cancel_action", "x")
        self.cancel_action.triggered.connect(self.cancel_recording)
        self.cancel_action.setEnabled(getattr(self, "is_recording", False))

        tray_menu.addSeparator()

        # --- Diagnostics / links ---
        # Add "Open Log File" action
        self.open_log_action = _add(_sp.SP_FileDialogInfoView, "tray_log_action", "l")
        self.open_log_action.triggered.connect(self.open_log_file)
        # Will be enabled/disabled based on log file existence

        # Add GitHub link
        github_action = _add(_sp.SP_DriveNetIcon, "menu_help_github", "g")
        github_action.triggered.connect(self.open_github_link)

        tray_menu.addSeparator()

        # --- App lifecycle ---
        # Self-update from git — only meaningful for a source checkout, so the entry is
        # omitted entirely for a frozen build (or a directory that isn't a git working tree).
        if getattr(self, "_updater", None) is None:
            self._updater = GitUpdater(self)
            self._updater.availability_changed.connect(lambda _available: self._apply_update_indicator())
            self._updater.pull_finished.connect(self._on_git_update_finished)
            self._updater.pull_failed_to_start.connect(self._on_git_update_failed_to_start)
        self.update_action: Optional[QAction] = None
        if self._updater.root:
            self.update_action = _add(_sp.SP_ArrowDown, "tray_update_action", "u")
            self.update_action.triggered.connect(self.check_for_updates)
            tray_menu.update_action = self.update_action
            # A rebuild (e.g. language change) must not lose an already-detected update — re-apply
            # the green-dot indicator from the updater's state.
            self._apply_update_indicator()

        restart_action = _add(_sp.SP_BrowserReload, "tray_restart_action", "n")
        restart_action.triggered.connect(self.restart_app)

        quit_action = _add(_sp.SP_DialogCloseButton, "tray_quit_action", "q")
        quit_action.triggered.connect(self.quit_app)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

        # Connect activation signal for left- and double-click handling.
        self.tray_icon.activated.connect(self.on_tray_icon_activated)

        # Initial state update for log file action
        self.update_logfile_menu_action()
        self.update_play_last_recording_action()

        # Kick off the periodic background update check (source checkouts only).
        self._updater.start_watching()

    def on_tray_icon_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        """
        Handle left-clicks and double-clicks on the system tray icon.

        Args:
            reason (QSystemTrayIcon.ActivationReason): The reason for the activation.
        """
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            # Windows emits another Trigger when the second button press of a double-click is
            # released. Ignore that release so the menu does not appear after copying.
            if time.monotonic() < self._ignore_tray_trigger_until:
                return

            if self.config.get("systray_double_click_copy", True):
                self._tray_single_click_timer.start(QApplication.doubleClickInterval())
            else:
                self._show_tray_menu()
        elif reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self._tray_single_click_timer.stop()
            self._ignore_tray_trigger_until = (
                time.monotonic() + QApplication.doubleClickInterval() / 1000.0
            )
            if self.config.get("systray_double_click_copy", True):
                logging.debug("Tray icon double-clicked, copying last transcription.")
                self.copy_last_transcription_to_clipboard()

    def _show_tray_menu(self) -> None:
        """Open the tray context menu at the current pointer position."""
        tray_menu = getattr(self, "tray_menu", None)
        if tray_menu is not None:
            tray_menu.popup(QCursor.pos())

    def update_play_last_recording_action(self) -> None:
        """Updates the enabled/disabled state of the 'Play/Re-transcribe Last Recording' actions."""
        exists = self.recordings.exists()
        if hasattr(self, 'play_action'):
            self.play_action.setEnabled(exists)
            if hasattr(self, 'play_last_recording_action'): # Also update the main menu action
                self.play_last_recording_action.setEnabled(exists)
        if hasattr(self, 'retranscribe_action'):
            self.retranscribe_action.setEnabled(exists)

    def _get_latest_recording_path(self) -> Optional[str]:
        """Return the path of the most recent recording, or None if none exists."""
        return self.recordings.latest()

    def retranscribe_last_recording(self) -> None:
        """Re-transcribes the most recent recording without recording again."""
        if not transcription_configured(self.config):
            self.show_tray_balloon(self.translator.tr("recording_no_api_keys"), BALLOON_INFO_MS)
            self.show_settings_window()
            return

        latest = self._get_latest_recording_path()
        if not latest or not os.path.isfile(latest):
            self.show_tray_balloon(self.translator.tr("no_recording_found_message"), BALLOON_SHORT_MS)
            self.update_play_last_recording_action()
            return

        logging.info(f"Re-transcribing last recording: {latest}")
        # No fresh selection context on a manual re-transcribe of an existing file.
        # Result goes to the clipboard (the user hasn't focused a text field).
        self.current_transcription_context = ""
        self.start_transcription_worker(latest, output_mode="clipboard")

    def transcribe_audio_file(self) -> None:
        """Pick an audio (or, with ffmpeg, video) file and transcribe it; result goes to the clipboard."""
        if not transcription_configured(self.config):
            self.show_tray_balloon(self.translator.tr("recording_no_api_keys"), BALLOON_INFO_MS)
            self.show_settings_window()
            return

        # With ffmpeg available we can also accept video containers (audio is extracted first).
        ffmpeg_exe = resolve_ffmpeg(self.config.get("ffmpeg_path", ""))
        audio_globs = "*.mp3 *.ogg *.wav *.m4a *.flac *.webm *.mp4 *.mpga *.mpeg"
        audio_filter = f"{self.translator.tr('audio_filter_label')} ({audio_globs})"
        all_files_filter = f"{self.translator.tr('all_files_filter_label')} (*)"
        if ffmpeg_exe:
            video_globs = " ".join(f"*{ext}" for ext in sorted(VIDEO_EXTENSIONS))
            media_filter = f"{self.translator.tr('media_filter_label')} ({audio_globs} {video_globs})"
            file_filter = f"{media_filter};;{audio_filter};;{all_files_filter}"
        else:
            file_filter = f"{audio_filter};;{all_files_filter}"

        # Reopen in the folder used last time (if it still exists), otherwise let Qt decide.
        start_dir = self.config.get("last_transcribe_dir", "")
        if start_dir and not os.path.isdir(start_dir):
            start_dir = ""

        paths, _ = QFileDialog.getOpenFileNames(
            None,
            self.translator.tr("transcribe_file_dialog_title"),
            start_dir,
            file_filter,
        )
        if not paths:
            return

        # Remember the folder of the picked files for next time.
        chosen_dir = os.path.dirname(paths[0])
        if chosen_dir and chosen_dir != self.config.get("last_transcribe_dir", ""):
            self.config["last_transcribe_dir"] = chosen_dir
            self.save_config()

        # Guard: a video was picked but no ffmpeg is configured — point the user at the setting.
        if not ffmpeg_exe and any(is_video_file(p) for p in paths):
            self.show_tray_balloon(self.translator.tr("video_needs_ffmpeg_message"), BALLOON_WARNING_MS)
            self.show_settings_window()
            return

        self.current_transcription_context = ""
        if len(paths) == 1:
            logging.info(f"Transcribing selected file: {paths[0]}")
            self.start_transcription_worker(paths[0], output_mode="clipboard")
        else:
            # Multiple files: transcribe sequentially and join the results with blank lines.
            logging.info("Transcribing %d selected files as a batch.", len(paths))
            self._start_batch_transcription(paths)

    def restart_app(self) -> None:
        """Relaunch WhisperTyper: spawn a fresh instance (``-r`` reclaims the lock), then exit.

        Our own global listeners + background capture are stopped up-front so the low-level
        keyboard hook is already gone before the new instance's ``-r`` hard-stops this one —
        an orphaned hook would otherwise cause system-wide input lag.
        """
        if getattr(sys, "frozen", False):
            program, arguments = sys.executable, ["-r"]
        else:
            program, arguments = sys.executable, [os.path.abspath(sys.argv[0]), "-r"]

        logging.info(f"Restarting application: {program} {arguments}")
        try:
            self._stop_hotkey_listeners()
            self._stop_background_audio_capture()
        except Exception:
            pass

        started = QProcess.startDetached(program, arguments)
        if not started:
            logging.error("Failed to launch a new instance for restart.")
            self.show_tray_balloon(self.translator.tr("restart_failed_message"), BALLOON_INFO_MS)
            return

        app = QApplication.instance()
        if app is not None:
            app.quit()

    # --- Self-update (git pull) ---------------------------------------------------------
    def check_for_updates(self) -> None:
        """Ask for confirmation, then pull the latest code from git for this source checkout.

        Only wired up when running from a git working tree (see ``init_tray_icon``). The pull
        runs asynchronously; ``_on_git_update_finished`` shows the result and offers a restart.
        """
        root = self._updater.root
        if not root:
            return
        if self._updater.pulling:
            self.show_tray_balloon(self.translator.tr("update_running_message"), BALLOON_SHORT_MS)
            return
        if not git_available():
            QMessageBox.warning(self, self.translator.tr("update_dialog_title"),
                                self.translator.tr("update_git_missing_text"))
            return
        reply = QMessageBox.question(
            self,
            self.translator.tr("update_dialog_title"),
            self.translator.tr("update_confirm_text", path=root),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.show_tray_balloon(self.translator.tr("update_running_message"), 0, spinner=True)
        self._updater.pull()

    def _on_git_update_failed_to_start(self) -> None:
        """Git vanished between the availability check and the pull."""
        self.hide_tray_balloon()
        QMessageBox.warning(self, self.translator.tr("update_dialog_title"),
                            self.translator.tr("update_git_missing_text"))

    def _on_git_update_finished(self, succeeded: bool, changed: bool, output: str) -> None:
        """Report the git pull result and, when new code arrived, offer to restart the app."""
        self.hide_tray_balloon()
        title = self.translator.tr("update_dialog_title")
        details = output or self.translator.tr("update_no_output")
        if not succeeded:
            QMessageBox.warning(self, title, self.translator.tr("update_failed_text", output=details))
            return
        if not changed:
            # Already up to date: nothing to restart for.
            QMessageBox.information(self, title, self.translator.tr("update_up_to_date_text"))
            return
        reply = QMessageBox.question(
            self, title, self.translator.tr("update_success_text", output=details),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.restart_app()

    def _apply_update_indicator(self) -> None:
        """Reflect the updater's availability on the tray entry label and its green dot."""
        action = getattr(self, "update_action", None)
        menu = getattr(self, "tray_menu", None)
        if action is None or menu is None:
            return
        available = self._updater.available
        action.setText(self.translator.tr("tray_update_available_action" if available else "tray_update_action"))
        menu.update_available = available
        menu.update()

    def show_settings_window(self) -> None:
        """Brings the settings window reliably to the foreground and logs its geometry."""
        t0 = time.perf_counter()
        try:
            if self.isMinimized():
                self.showNormal()
            else:
                self.show()
            self.raise_()
            self.activateWindow()
            t_shown = time.perf_counter()
            # show() returns before the first paint happens; flush pending events so the timing
            # below reflects the real time until the window is actually rendered.
            QApplication.processEvents()
            t_painted = time.perf_counter()
            logging.info(
                "show_settings_window timing: show/raise/activate=%.0fms, first_paint_flush=%.0fms, total=%.0fms",
                (t_shown - t0) * 1000, (t_painted - t_shown) * 1000, (t_painted - t0) * 1000,
            )

            geo = self.geometry()
            frame = self.frameGeometry()
            screen = QApplication.screenAt(frame.center()) or QApplication.primaryScreen()
            screen_info = ""
            if screen:
                sg = screen.availableGeometry()
                screen_info = (f", screen='{screen.name()}' "
                               f"available=({sg.x()},{sg.y()},{sg.width()}x{sg.height()})")
            logging.info(
                "Settings window shown: "
                f"visible={self.isVisible()}, active={self.isActiveWindow()}, minimized={self.isMinimized()}, "
                f"geometry=({geo.x()},{geo.y()},{geo.width()}x{geo.height()}), "
                f"frame=({frame.x()},{frame.y()},{frame.width()}x{frame.height()})"
                f"{screen_info}"
            )
        except Exception as e:
            logging.error(f"Failed to show settings window: {e}")

        # Refresh the mic list AFTER the window is visible, on the next event-loop tick, so a slow
        # PyAudio enumeration never blocks the window from appearing. The device dropdown lives on
        # the General tab (not the default tab), so it needn't be ready the instant settings open.
        if hasattr(self, "_populate_input_device_selector"):
            QTimer.singleShot(0, self._populate_input_device_selector)

    def show_transformations_settings(self) -> None:
        """Open the settings window directly on the transformations templates tab."""
        self.tabs.setCurrentWidget(self.post_rephrasing_tab)
        self.show_settings_window()

    def _get_app_icon(self) -> QIcon:
        """Return the platform-appropriate application icon."""
        return app_icon()
