"""System-tray icon and menu: actions, live recording meter and the self-update entry."""
from __future__ import annotations

import logging
import os
import time
from typing import Callable, Dict, Optional, Protocol, Tuple

from PyQt6.QtCore import QObject, QPoint, QTimer
from PyQt6.QtGui import QAction, QCursor, QIcon
from PyQt6.QtWidgets import QApplication, QFileDialog, QLabel, QStyle, QSystemTrayIcon, QWidget

from app import __version__
from app.bootstrap import requirements_install_command
from app.context import AppContext
from app.controllers.file_actions import FileActions
from app.controllers.local_engine import LOCAL_ENGINE_SUPPORTED, UNAVAILABLE, LocalEngineController
from app.controllers.recording import RecordingController
from app.controllers.transcription import TranscriptionPipeline
from app.core.api_keys import transcription_configured
from app.core.audio_formats import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, requires_ffmpeg
from app.core.constants import APP_DATA_DIR
from app.core.env import is_MACOS
from app.core.hotkeys import pretty_hotkey
from app.local_engine.supervisor import OFF, READY, EngineStatus
from app.services import bad_recordings
from app.services.ffmpeg import resolve_ffmpeg
from app.services.gitutil import git_available, release_changes
from app.services.releases import Release
from app.services.updater import GitUpdater
from app.ui import macos_icons, message_dialog
from app.ui.about_dialog import show_about
from app.ui.durations import BALLOON_INFO_MS, BALLOON_SHORT_MS, BALLOON_WARNING_MS
from app.ui.line_icons import line_icon
from app.ui.local_engine_tab import status_text
from app.ui.tray_icons import app_icon, recording_level_icon, tray_icon
from app.ui.tray_menu import BadgeTrayMenu

#: Refresh interval of the recording level meter in the tray icon.
_METER_INTERVAL_MS = 80
#: Modern design: the line icon (``resources/icons``) of each tray entry, by its label key.
_MODERN_ICONS = {
    "tray_settings_action": "gear",
    "menu_file_open_config": "file",
    "tray_edit_transformations_action": "sparkles",
    "tray_copy_action": "copy",
    "tray_retranscribe_action": "refresh",
    "tray_transcribe_file_action": "folder",
    "tray_play_action": "play",
    "tray_mark_bad_action": "flag",
    "tray_cancel_action": "close",
    "tray_log_action": "log",
    "menu_help_github": "code",
    "tray_update_action": "download",
    "tray_about_action": "info",
    "tray_restart_action": "restart",
    "tray_quit_action": "quit",
}


class SettingsWindowLike(Protocol):
    """The settings-window operations the tray menu opens (see ``app.ui.settings.window``)."""

    def show_window(self) -> None:
        """Bring the settings window to the foreground."""

    def show_transformations(self) -> None:
        """Open the settings window on the transformations tab."""

    def theme_palette(self) -> Dict[str, str]:
        """Colours of the active theme."""

    def is_modern(self) -> bool:
        """Whether the window shows the modern design (the tray menu matches it)."""

    def show_local_engine(self) -> None:
        """Open the settings window on the local transcription page."""


class TrayController(QObject):
    """Own the tray icon, its context menu and the recording-state icon."""

    def __init__(self, ctx: AppContext, window: QWidget, settings: SettingsWindowLike, *,
                 recording: RecordingController, pipeline: TranscriptionPipeline, files: FileActions,
                 restart: Callable[[], None], quit_app: Callable[[], None],
                 local_engine: Optional[LocalEngineController] = None) -> None:
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
        # Whether the running update is a forced one (its result dialog reads differently).
        self._update_forced = False
        # The user asked for an update check; its answer is reported in a dialog.
        self._update_check_requested = False
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
        self.updater.checked.connect(self._on_update_checked)
        self.update_action: Optional[QAction] = None
        self._header_status: Optional[QLabel] = None
        self._tooltip = ""
        self._local_engine = local_engine
        self._local_menu: Optional[BadgeTrayMenu] = None
        self._local_actions: Dict[str, QAction] = {}
        if local_engine is not None:
            local_engine.status_changed.connect(lambda _status: self._refresh_local_menu())
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
        if active and self._settings.is_modern() and not is_MACOS:
            return  # the modern design keeps the app icon; the recording overlay shows the level
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
        """Tooltip of the tray icon (also refreshes the modern menu header's hotkey)."""
        self._tooltip = text
        if self.tray_icon is not None:
            self.tray_icon.setToolTip(text)
        self._update_header()

    def _menu_icon_color(self) -> str:
        """Tint of the modern menu's line icons in the active theme."""
        colors = self._settings.theme_palette()
        return colors.get("menu_icon", colors["muted"])

    def _update_header(self) -> None:
        """Modern design: the menu header shows the current recording hotkey."""
        if self._header_status is not None:
            self._header_status.setText(pretty_hotkey(self._ctx.config.get("hotkey", "")))

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
        elif self._settings.is_modern():
            # The modern design always shows the microphone tile, also while recording.
            self._idle_icon = tray_icon()
            if self._idle_icon.isNull():
                self._idle_icon = standard_icon
        else:
            self._idle_icon = standard_icon
        self.tray_icon.setIcon(self._idle_icon)
        self.tray_icon.setToolTip(self._tooltip)  # a rebuild (language, theme) must keep it

        # QSystemTrayIcon.setContextMenu() does not take ownership: the menu needs a parent and
        # a Python reference, otherwise it is collected and its actions never fire.
        modern = self._settings.is_modern()
        tray_menu = BadgeTrayMenu(window, self._settings.theme_palette, modern=modern)
        self.tray_menu = tray_menu
        tr = self._ctx.tr
        sp = QStyle.StandardPixmap
        self._header_status = None
        if modern:
            self._header_status = tray_menu.add_header(app_icon(), "WhisperTyper", "", __version__)
            self._update_header()
            tray_menu.addSeparator()

        def add(pixmap: QStyle.StandardPixmap, label_key: str, key: str, handler: Callable[[], None]) -> QAction:
            # BadgeTrayMenu paints <key> as a badge and triggers the action when it is pressed.
            # Letters are fixed in code so they stay unique regardless of the UI language.
            icon = (line_icon(_MODERN_ICONS[label_key], self._menu_icon_color()) if modern
                    else style.standardIcon(pixmap))
            action = tray_menu.addAction(icon, tr(label_key))
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
        self.mark_bad_action = add(sp.SP_MessageBoxWarning, "tray_mark_bad_action", "b", self.mark_last_recording_bad)
        # Discards the running recording (no transcription, no clipboard change, the stored
        # "last recording" stays). Enabled only while recording.
        self.cancel_action = add(sp.SP_DialogCancelButton, "tray_cancel_action", "x", self._recording.cancel)
        self.cancel_action.setEnabled(self._recording.is_recording)
        self._build_local_menu(tray_menu, modern)
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
        add(sp.SP_MessageBoxInformation, "tray_about_action", "a", lambda: show_about(self._window, self._ctx.tr))
        tray_menu.addSeparator()
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
        self.mark_bad_action.setEnabled(exists)
        self.open_log_action.setEnabled(self._files.log_file_exists())
        self._refresh_local_menu()

    # --- Local transcription submenu ---------------------------------------------------
    def _build_local_menu(self, tray_menu: BadgeTrayMenu, modern: bool) -> None:
        """Status and actions of the local engine; shown only while it is enabled."""
        self._local_menu, self._local_actions = None, {}
        engine = self._local_engine
        if engine is None or not LOCAL_ENGINE_SUPPORTED:
            return
        tr = self._ctx.tr
        menu = BadgeTrayMenu(self._window, self._settings.theme_palette, modern=modern)
        menu.setTitle(tr("tray_local_menu"))
        if modern:
            menu.setIcon(line_icon("chip", self._menu_icon_color()))
        status = menu.addAction("")
        assert status is not None
        status.setEnabled(False)
        menu.addSeparator()
        actions = {"status": status}
        for key, label_key, icon, handler in (
            ("load", "local_load_button", "download", self._toggle_local_engine),
            ("self_test", "local_self_test_button", "check", engine.self_test),
            ("restart", "local_restart_button", "restart", engine.restart),
            ("copy", "local_server_copy_button", "copy", self._copy_local_endpoint),
            ("log", "local_log_button", "log", self._files.open_engine_log_file),
            ("settings", "tray_local_settings", "gear", self._settings.show_local_engine),
        ):
            action = menu.addAction(line_icon(icon, self._menu_icon_color()) if modern else QIcon(), tr(label_key))
            assert action is not None
            action.triggered.connect(lambda _checked=False, handler=handler: handler())
            actions[key] = action
        self._local_actions = actions
        self._local_menu = menu
        tray_menu.addSeparator()
        menu_action = tray_menu.addMenu(menu)
        assert menu_action is not None
        self._local_menu_action = menu_action

    def _refresh_local_menu(self) -> None:
        """Follow the engine's status and the saved settings."""
        engine, menu = self._local_engine, self._local_menu
        if engine is None or menu is None or not self._local_actions:
            return
        tr = self._ctx.tr
        status: EngineStatus = engine.status
        self._local_menu_action.setVisible(engine.enabled())
        self._local_actions["status"].setText(status_text(self._ctx.translator, status, engine.devices or []))
        running = status.state not in (OFF, UNAVAILABLE)
        self._local_actions["load"].setText(tr("local_unload_button" if running else "local_load_button"))
        self._local_actions["self_test"].setEnabled(status.state == READY)
        self._local_actions["copy"].setVisible(engine.server_enabled())

    def _toggle_local_engine(self) -> None:
        engine = self._local_engine
        if engine is None:
            return
        if engine.status.state not in (OFF, UNAVAILABLE):
            engine.stop()
        else:
            engine.restart()

    def _copy_local_endpoint(self) -> None:
        if self._local_engine is not None:
            clipboard = QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(self._local_engine.endpoint())
                self._ctx.notifier.show(self._ctx.tr("local_server_endpoint_copied"), BALLOON_SHORT_MS)

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

    def mark_last_recording_bad(self) -> None:
        """Copy the latest recording with its sent request and raw transcript for later analysis."""
        latest = self._ctx.recordings.latest()
        if not latest or not os.path.isfile(latest):
            self._ctx.notifier.show(self._ctx.tr("no_recording_found_message"), BALLOON_SHORT_MS)
            self.refresh_actions()
            return
        dest = self._ctx.config.get("bad_recordings_dir") or os.path.join(APP_DATA_DIR, "bad_recordings")
        try:
            saved = bad_recordings.save(dest, latest, self._pipeline.recording_request(latest))
        except OSError as error:
            logging.error("bad_recording_save_failed dest=%s error=%s", dest, error)
            self._ctx.notifier.show(self._ctx.tr("recording_mark_bad_failed_message", error=error), BALLOON_WARNING_MS)
            return
        logging.info("bad_recording_saved path=%s", saved)
        self._ctx.notifier.show(self._ctx.tr("recording_marked_bad_message", path=saved), BALLOON_INFO_MS)

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

    # --- Self-update (release tags) -------------------------------------------------------
    def check_for_updates(self) -> None:
        """Offer a known newer release, or look for one and report the answer."""
        if not self.updater.root:
            return
        tr = self._ctx.tr
        if self.updater.pulling:
            self._ctx.notifier.show(tr("update_running_message"), BALLOON_SHORT_MS)
            return
        if not git_available():
            self._update_message("update_git_missing_title", tr("update_git_missing_text"), message_dialog.WARNING)
            return
        if self.updater.latest is not None:
            self._offer_update(self.updater.latest)
            return
        self._update_check_requested = True
        self._ctx.notifier.show(tr("update_checking_message"), 0, spinner=True)
        self.updater.check_now()

    def _update_message(self, title_key: str, text: str, kind: Tuple[str, str], **kwargs: str) -> None:
        """A self-update message with a single close button."""
        tr = self._ctx.tr
        message_dialog.inform(self._window, tr(title_key, **kwargs), text, tr("update_close_button"), kind)

    def _on_update_checked(self, reachable: bool) -> None:
        """Answer a check the user asked for; background checks only update the menu entry."""
        if not self._update_check_requested:
            return
        self._update_check_requested = False
        self._ctx.notifier.hide()
        tr = self._ctx.tr
        if not reachable:
            self._update_message("update_check_failed_title", tr("update_check_failed_text"), message_dialog.WARNING)
        elif self.updater.latest is None:
            # After an update (before the restart) ``current`` is the installed release.
            current = self.updater.current.version if self.updater.current else __version__
            self._update_message("update_up_to_date_title", tr("update_up_to_date_text", version=current),
                                 message_dialog.SUCCESS)
        else:
            self._offer_update(self.updater.latest)

    def _offer_update(self, release: Release) -> None:
        """Ask whether to update to ``release``; its changes can be unfolded below the question."""
        tr = self._ctx.tr
        update = tr("update_install_button")
        choice = message_dialog.show_dialog(
            self._window, tr("update_available_title", version=release.version),
            tr("update_confirm_text", version=release.version, current=__version__),
            [(tr("update_cancel_button"), message_dialog.SECONDARY), (update, message_dialog.PRIMARY)],
            message_dialog.UPDATE, details=release_changes(self.updater.root or "", release.tag) or "",
            details_captions=(tr("update_changes_button"), tr("update_changes_hide_button")),
            window_title=tr("update_dialog_title"))
        if choice != update or self.updater.pulling:
            return
        self._ctx.notifier.show(tr("update_running_message"), 0, spinner=True)
        self.updater.pull()

    def _on_update_failed_to_start(self) -> None:
        """Git vanished between the availability check and the update."""
        self._ctx.notifier.hide()
        self._update_message("update_git_missing_title", self._ctx.tr("update_git_missing_text"),
                             message_dialog.WARNING)

    def _force_update(self) -> None:
        """After an explicit confirmation, replace local changes with the newest release."""
        tr = self._ctx.tr
        if self.updater.pulling or self.updater.latest is None:
            return
        version = self.updater.latest.version
        if not message_dialog.ask(self._window, tr("update_force_title", version=version),
                                  tr("update_force_confirm_text", version=version, path=self.updater.root or ""),
                                  tr("update_force_confirm_button"), tr("update_cancel_button"),
                                  message_dialog.WARNING, window_title=tr("update_dialog_title")):
            return
        self._update_forced = True
        self._ctx.notifier.show(tr("update_running_message"), 0, spinner=True)
        self.updater.force_update()

    def _on_update_finished(self, succeeded: bool, changed: bool, output: str) -> None:
        """Report the update result; offer a forced update after a failure and a restart after new code."""
        self._ctx.notifier.hide()
        forced, self._update_forced = self._update_forced, False
        tr = self._ctx.tr
        window_title = tr("update_dialog_title")
        details = output or tr("update_no_output")
        details_captions = (tr("update_details_button"), tr("update_details_hide_button"))
        version = self.updater.target.version if self.updater.target else ""
        close = (tr("update_close_button"), message_dialog.SECONDARY if not forced else message_dialog.PRIMARY)
        if not succeeded:
            force = tr("update_force_button")
            message = tr("update_failed_text", version=version)
            if not forced:
                message += "\n\n" + tr("update_force_hint", version=version)
            buttons = [close] if forced else [close, (force, message_dialog.DANGER)]
            if message_dialog.show_dialog(self._window, tr("update_failed_title"), message, buttons,
                                          message_dialog.ERROR, details, details_captions, window_title) == force:
                self._force_update()
            return
        if not changed:
            self._update_message("update_up_to_date_title", tr("update_unchanged_text", version=version),
                                 message_dialog.SUCCESS)
            return
        restart = tr("update_restart_button")
        message = tr("update_success_text")
        if forced:
            message += "\n\n" + tr("update_force_success_hint")
        if self.updater.requirements_changed:
            # The new code may need packages the running interpreter lacks; it could not start.
            message += "\n\n" + tr("update_requirements_hint", command=requirements_install_command())
        if message_dialog.show_dialog(self._window, tr("update_success_title", version=version), message,
                                      [(tr("update_later_button"), message_dialog.SECONDARY),
                                       (restart, message_dialog.PRIMARY)],
                                      message_dialog.SUCCESS, details, details_captions, window_title) == restart:
            self._restart()

    def _apply_update_indicator(self) -> None:
        """Reflect the updater's availability on the tray entry label and its green dot."""
        if self.update_action is None or self.tray_menu is None:
            return
        latest = self.updater.latest
        available = latest is not None
        self.update_action.setText(self._ctx.tr("tray_update_available_action", version=latest.version)
                                   if latest is not None else self._ctx.tr("tray_update_action"))
        if self._header_status is not None:
            colors = self._settings.theme_palette()
            self.update_action.setIcon(line_icon("download", colors["accent"] if available else self._menu_icon_color()))
        self.tray_menu.update_available = available
        self.tray_menu.update()
