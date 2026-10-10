"""System-tray icon and menu: actions, live recording meter and the git self-update entry."""
from __future__ import annotations

import logging
import os
import time
from typing import Callable, Dict, Optional, Protocol, Tuple

from PyQt6.QtCore import QObject, QPoint, QTimer
from PyQt6.QtGui import QAction, QCursor, QIcon
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox, QStyle, QSystemTrayIcon, QWidget

from app.context import AppContext
from app.controllers.file_actions import FileActions
from app.controllers.recording import RecordingController
from app.controllers.transcription import TranscriptionPipeline
from app.core.api_keys import transcription_configured
from app.core.audio_formats import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, requires_ffmpeg
from app.core.env import is_MACOS
from app.services.ffmpeg import resolve_ffmpeg
from app.services.gitutil import git_available
from app.services.updater import GitUpdater
from app.ui import macos_icons
from app.ui.durations import BALLOON_INFO_MS, BALLOON_SHORT_MS, BALLOON_WARNING_MS
from app.ui.tray_icons import app_icon, recording_level_icon
from app.ui.tray_menu import BadgeTrayMenu

#: Refresh interval of the recording level meter in the tray icon.
_METER_INTERVAL_MS = 80


class SettingsWindowLike(Protocol):
    """The settings-window operations the tray menu opens (see ``app.ui.settings.window``)."""

    def show_window(self) -> None:
        """Bring the settings window to the foreground."""

    def show_transformations(self) -> None:
        """Open the settings window on the transformations tab."""

    def theme_palette(self) -> Dict[str, str]:
        """Colours of the active theme."""


class TrayController(QObject):
    """Own the tray icon, its context menu and the recording-state icon."""

    def __init__(self, ctx: AppContext, window: QWidget, settings: SettingsWindowLike, *,
                 recording: RecordingController, pipeline: TranscriptionPipeline, files: FileActions,
                 restart: Callable[[], None], quit_app: Callable[[], None]) -> None:
        """Icon, menu and dialogs belong to ``window`` (the settings window) so they share its theme."""
        super().__init__()
        self._ctx = ctx
        self._window = window
        self._settings = settings
        self._recording = recording
        self._pipeline = pipeline
        self._files = files
        self._restart = restart
        self._quit = quit_app
        self.tray_icon: Optional[QSystemTrayIcon] = None
        self.tray_menu: Optional[BadgeTrayMenu] = None
        self._idle_icon = QIcon()
        self._recording_icon_cache: Dict[Tuple[int, bool], QIcon] = {}
        self._meter_timer = QTimer(self)
        self._meter_timer.setInterval(_METER_INTERVAL_MS)
        self._meter_timer.timeout.connect(self._update_recording_icon)
        # Delay a left-click just long enough to distinguish it from the double-click gesture.
        # Without that arbitration, opening the menu on the first release would prevent the
        # second click from reaching the tray icon and break double-click-to-copy.
        self._single_click_timer = QTimer(self)
        self._single_click_timer.setSingleShot(True)
        self._single_click_timer.timeout.connect(self._show_menu)
        self._ignore_trigger_until = 0.0
        self.updater = GitUpdater(self)
        self.updater.availability_changed.connect(lambda _available: self._apply_update_indicator())
        self.updater.pull_finished.connect(self._on_update_finished)
        self.updater.pull_failed_to_start.connect(self._on_update_failed_to_start)
        self.update_action: Optional[QAction] = None
        recording.state_changed.connect(self.set_recording)
        ctx.files_changed.connect(self.refresh_actions)

    # --- Icon ----------------------------------------------------------------------------
    def anchor(self) -> Optional[QPoint]:
        """Centre of the tray icon on screen, if the platform reports it."""
        if self.tray_icon is None:
            return None
        geometry = self.tray_icon.geometry()
        return geometry.center() if geometry.isValid() else None

    def set_recording(self, active: bool) -> None:
        """Animate the level meter while recording; restore the idle icon otherwise."""
        self.cancel_action.setEnabled(active)
        if active:
            self._recording.latest_audio_level = 0.0
            self._update_recording_icon()
            if not self._meter_timer.isActive():
                self._meter_timer.start()
        else:
            self._set_idle_icon()

    def _set_idle_icon(self) -> None:
        """Restore the non-recording tray icon."""
        if self._meter_timer.isActive():
            self._meter_timer.stop()
        self._recording.latest_audio_level = 0.0
        if self.tray_icon is not None and not self._idle_icon.isNull():
            self.tray_icon.setIcon(self._idle_icon)

    def _update_recording_icon(self) -> None:
        """Refresh the tray icon with the latest measured input level."""
        if not self._recording.is_recording:
            self._set_idle_icon()
            return
        level = self._recording.level()
        icon = QIcon()
        if is_MACOS:
            icon = macos_icons.recording_tray_icon(level, app_icon())
        if icon.isNull():
            icon = recording_level_icon(level, self._recording_icon_cache)
        if self.tray_icon is not None:
            self.tray_icon.setIcon(icon)

    def set_tooltip(self, text: str) -> None:
        """Tooltip of the tray icon."""
        if self.tray_icon is not None:
            self.tray_icon.setToolTip(text)

    def hide(self) -> None:
        """Remove the icon at quit."""
        if self.tray_icon is not None:
            self.tray_icon.hide()

    # --- Menu ----------------------------------------------------------------------------
    def build(self) -> None:
        """(Re)create the tray icon and its menu in the current UI language."""
        self._single_click_timer.stop()
        # Fully dispose of a previous tray icon + menu before creating new ones: they are
        # parented to the window, so merely reassigning the attribute would leak them under
        # the Qt parent on every rebuild (e.g. each language change).
        if self.tray_icon is not None:
            self.tray_icon.hide()
            self.tray_icon.deleteLater()
        if self.tray_menu is not None:
            self.tray_menu.deleteLater()
            self.tray_menu = None

        window = self._window
        style = window.style()
        assert style is not None
        self.tray_icon = QSystemTrayIcon(window)
        standard_icon = style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        if is_MACOS:
            self._idle_icon = macos_icons.tray_icon()
            if self._idle_icon.isNull():
                self._idle_icon = app_icon()
            if self._idle_icon.isNull():
                self._idle_icon = standard_icon
        else:
            self._idle_icon = standard_icon
        self.tray_icon.setIcon(self._idle_icon)

        # QSystemTrayIcon.setContextMenu() does not take ownership: the menu needs a parent and
        # a Python reference, otherwise it is collected and its actions never fire.
        tray_menu = BadgeTrayMenu(window, self._settings.theme_palette)
        self.tray_menu = tray_menu
        tr = self._ctx.tr
        sp = QStyle.StandardPixmap

        def add(pixmap: QStyle.StandardPixmap, label_key: str, key: str, handler: Callable[[], None]) -> QAction:
            # BadgeTrayMenu paints <key> as a badge and triggers the action when it is pressed.
            # Letters are fixed in code so they stay unique regardless of the UI language.
            action = tray_menu.addAction(style.standardIcon(pixmap), tr(label_key))
            assert action is not None
            tray_menu.register_badge(key, action)
            action.triggered.connect(lambda _checked=False: handler())
            return action

        add(sp.SP_FileDialogDetailedView, "tray_settings_action", "k", self._settings.show_window)
        add(sp.SP_FileIcon, "menu_file_open_config", "o", self._files.open_config_file)
        add(sp.SP_FileDialogListView, "tray_edit_transformations_action", "t", self._settings.show_transformations)
        tray_menu.addSeparator()
        add(sp.SP_FileDialogContentsView, "tray_copy_action", "c", self._pipeline.copy_last_transcription)
        self.retranscribe_action = add(sp.SP_BrowserReload, "tray_retranscribe_action", "r",
                                       self.retranscribe_last_recording)
        add(sp.SP_DialogOpenButton, "tray_transcribe_file_action", "f", self.transcribe_audio_files)
        self.play_action = add(sp.SP_MediaPlay, "tray_play_action", "p", self._files.play_latest_recording)
        # Discards the running recording (no transcription, no clipboard change, the stored
        # "last recording" stays). Enabled only while recording.
        self.cancel_action = add(sp.SP_DialogCancelButton, "tray_cancel_action", "x", self._recording.cancel)
        self.cancel_action.setEnabled(self._recording.is_recording)
        tray_menu.addSeparator()
        self.open_log_action = add(sp.SP_FileDialogInfoView, "tray_log_action", "l", self._files.open_log_file)
        add(sp.SP_DriveNetIcon, "menu_help_github", "g", self._files.open_github_link)
        tray_menu.addSeparator()
        # Self-update from git is only meaningful for a source checkout.
        self.update_action = None
        if self.updater.root:
            self.update_action = add(sp.SP_ArrowDown, "tray_update_action", "u", self.check_for_updates)
            tray_menu.update_action = self.update_action
            # A rebuild must not lose an already-detected update.
            self._apply_update_indicator()
        add(sp.SP_BrowserReload, "tray_restart_action", "n", self._restart)
        add(sp.SP_DialogCloseButton, "tray_quit_action", "q", self._quit)

        # macOS opens a context menu natively on every click, on top of the popup from
        # _on_activated, which stacked two menus; there the popup alone handles both clicks.
        if not is_MACOS:
            self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()
        self.tray_icon.activated.connect(self._on_activated)
        self.refresh_actions()
        # Kick off the periodic background update check (source checkouts only).
        self.updater.start_watching()

    def refresh_actions(self) -> None:
        """Enable the recording and log actions only when their files exist."""
        if self.tray_menu is None:
            return
        exists = self._ctx.recordings.exists()
        self.play_action.setEnabled(exists)
        self.retranscribe_action.setEnabled(exists)
        self.open_log_action.setEnabled(self._files.log_file_exists())

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        """Left-click opens the menu; double-click copies the last transcription (if enabled)."""
        double_click_copy = self._ctx.config.get("systray_double_click_copy", True)
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            # Windows emits another Trigger when the second button press of a double-click is
            # released. Ignore that release so the menu does not appear after copying.
            if time.monotonic() < self._ignore_trigger_until:
                return
            if double_click_copy:
                self._single_click_timer.start(QApplication.doubleClickInterval())
            else:
                self._show_menu()
        elif reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self._single_click_timer.stop()
            self._ignore_trigger_until = time.monotonic() + QApplication.doubleClickInterval() / 1000.0
            if double_click_copy:
                logging.debug("Tray icon double-clicked, copying last transcription.")
                self._pipeline.copy_last_transcription()

    def _show_menu(self) -> None:
        """Open the tray context menu at the current pointer position."""
        if self.tray_menu is not None:
            self.tray_menu.popup(QCursor.pos())

    # --- File transcription --------------------------------------------------------------
    def _require_transcription_settings(self) -> bool:
        """Open the settings when no transcription request could be sent yet."""
        if transcription_configured(self._ctx.config):
            return True
        self._ctx.notifier.show(self._ctx.tr("recording_no_api_keys"), BALLOON_INFO_MS)
        self._settings.show_window()
        return False

    def retranscribe_last_recording(self) -> None:
        """Re-transcribe the most recent recording; the result goes to the clipboard."""
        if not self._require_transcription_settings():
            return
        latest = self._ctx.recordings.latest()
        if not latest or not os.path.isfile(latest):
            self._ctx.notifier.show(self._ctx.tr("no_recording_found_message"), BALLOON_SHORT_MS)
            self.refresh_actions()
            return
        logging.info(f"Re-transcribing last recording: {latest}")
        # No fresh selection context on a manual re-transcribe of an existing file.
        self._pipeline.selection_context = ""
        self._pipeline.start(latest, output_mode="clipboard")

    def transcribe_audio_files(self) -> None:
        """Pick audio (or, with ffmpeg, video) files and transcribe them; results go to the clipboard."""
        if not self._require_transcription_settings():
            return
        config, tr = self._ctx.config, self._ctx.tr
        # With ffmpeg available we can also accept video containers (audio is extracted first).
        ffmpeg_exe = resolve_ffmpeg(config.get("ffmpeg_path", ""))
        audio_globs = " ".join(f"*{ext}" for ext in sorted(AUDIO_EXTENSIONS))
        audio_filter = f"{tr('audio_filter_label')} ({audio_globs})"
        all_files_filter = f"{tr('all_files_filter_label')} (*)"
        if ffmpeg_exe:
            video_globs = " ".join(f"*{ext}" for ext in sorted(VIDEO_EXTENSIONS - AUDIO_EXTENSIONS))
            media_filter = f"{tr('media_filter_label')} ({audio_globs} {video_globs})"
            file_filter = f"{media_filter};;{audio_filter};;{all_files_filter}"
        else:
            file_filter = f"{audio_filter};;{all_files_filter}"

        # Reopen in the folder used last time (if it still exists), otherwise let Qt decide.
        start_dir = config.get("last_transcribe_dir", "")
        if start_dir and not os.path.isdir(start_dir):
            start_dir = ""
        paths, _ = QFileDialog.getOpenFileNames(None, tr("transcribe_file_dialog_title"), start_dir, file_filter)
        if not paths:
            return

        chosen_dir = os.path.dirname(paths[0])
        if chosen_dir and chosen_dir != config.get("last_transcribe_dir", ""):
            config["last_transcribe_dir"] = chosen_dir
            self._ctx.save_config()

        # Guard: a video the APIs reject was picked but no ffmpeg is configured — point at the setting.
        if not ffmpeg_exe and any(requires_ffmpeg(p) for p in paths):
            self._ctx.notifier.show(tr("video_needs_ffmpeg_message"), BALLOON_WARNING_MS)
            self._settings.show_window()
            return

        self._pipeline.selection_context = ""
        if len(paths) == 1:
            logging.info(f"Transcribing selected file: {paths[0]}")
            self._pipeline.start(paths[0], output_mode="clipboard")
        else:
            logging.info("Transcribing %d selected files as a batch.", len(paths))
            self._pipeline.start_batch(paths)

    # --- Self-update (git pull) ----------------------------------------------------------
    def check_for_updates(self) -> None:
        """Ask for confirmation, then pull the latest code from git for this source checkout."""
        root = self.updater.root
        if not root:
            return
        tr = self._ctx.tr
        if self.updater.pulling:
            self._ctx.notifier.show(tr("update_running_message"), BALLOON_SHORT_MS)
            return
        if not git_available():
            QMessageBox.warning(self._window, tr("update_dialog_title"), tr("update_git_missing_text"))
            return
        reply = QMessageBox.question(
            self._window, tr("update_dialog_title"), tr("update_confirm_text", path=root),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._ctx.notifier.show(tr("update_running_message"), 0, spinner=True)
        self.updater.pull()

    def _on_update_failed_to_start(self) -> None:
        """Git vanished between the availability check and the pull."""
        self._ctx.notifier.hide()
        QMessageBox.warning(self._window, self._ctx.tr("update_dialog_title"), self._ctx.tr("update_git_missing_text"))

    def _on_update_finished(self, succeeded: bool, changed: bool, output: str) -> None:
        """Report the git pull result and, when new code arrived, offer to restart the app."""
        self._ctx.notifier.hide()
        tr = self._ctx.tr
        title = tr("update_dialog_title")
        details = output or tr("update_no_output")
        if not succeeded:
            QMessageBox.warning(self._window, title, tr("update_failed_text", output=details))
            return
        if not changed:
            QMessageBox.information(self._window, title, tr("update_up_to_date_text"))
            return
        reply = QMessageBox.question(
            self._window, title, tr("update_success_text", output=details),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._restart()

    def _apply_update_indicator(self) -> None:
        """Reflect the updater's availability on the tray entry label and its green dot."""
        if self.update_action is None or self.tray_menu is None:
            return
        available = self.updater.available
        self.update_action.setText(self._ctx.tr("tray_update_available_action" if available else "tray_update_action"))
        self.tray_menu.update_available = available
        self.tray_menu.update()
