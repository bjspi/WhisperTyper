"""TrayMixin — system-tray icon, context menu and recording-state icon."""
from __future__ import annotations

import logging
import os
import sys
import time
from typing import List, Optional

from PyQt6.QtCore import QPointF, QRect, QRectF, Qt, QTimer
from PyQt6.QtGui import QAction, QColor, QIcon, QKeyEvent, QPainter, QPaintEvent, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMenu,
    QStyle,
    QSystemTrayIcon,
    QWidget,
)

from app.core.env import is_MACOS
from app.core.ffmpeg import VIDEO_EXTENSIONS, is_video_file, resolve_ffmpeg
from app.core.gitutil import (
    count_behind_upstream,
    current_head,
    find_git_root,
    git_available,
)
from app.core.paths import resource_path

_BADGE_BOX = 18   # accelerator badge square size in px
_BADGE_MARGIN = 8  # gap between the badge and the item's right edge

# Background update watcher (source checkouts only). First fetch runs shortly after launch so
# startup isn't slowed; after that it polls periodically until an update is found, then stops.
# Offline / failed fetches are ignored and simply retried on the next tick.
_UPDATE_CHECK_INITIAL_DELAY_MS = 20_000       # ~20 s after launch
_UPDATE_CHECK_INTERVAL_MS = 30 * 60 * 1000    # then every 30 minutes


class _BadgeTrayMenu(QMenu):
    """Tray context menu that paints a rounded-square accelerator badge on the far right of
    each item, and triggers that item directly when its letter is pressed while the menu is open.

    The badge is drawn in ``paintEvent`` (on top of the normal QSS render) rather than via a
    custom ``QStyle``: wrapping the widget's stylesheet style in a QProxyStyle recurses at the
    C++ level and hard-crashes on first paint. Painting over the finished menu is crash-safe and
    keeps the full QSS look (panel, rounded corners, full-width highlight, icon indent) intact.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._app = parent  # WhisperTyperApp — read live theme palette via getattr
        self._badge_keys: dict[str, QAction] = {}

    def register_badge(self, key: str, action: QAction) -> None:
        """Map a lowercase accelerator letter to the action it triggers and stamp it for paint."""
        key = key.lower()
        self._badge_keys[key] = action
        action.setData(key)  # read back in paintEvent to render the badge

    def keyPressEvent(self, event: QKeyEvent | None) -> None:  # noqa: N802 (Qt override)
        if event is not None:
            action = self._badge_keys.get((event.text() or "").lower())
            if action is not None and action.isEnabled():
                self.close()
                action.trigger()
                return
        super().keyPressEvent(event)

    def _palette(self) -> dict:
        colors = getattr(self._app, "_theme_palette", None)
        if colors:
            return colors
        from app.ui import theme
        return theme.palette(False)

    def paintEvent(self, event: QPaintEvent | None) -> None:  # noqa: N802 (Qt override)
        super().paintEvent(event)  # normal (QSS) menu render first
        colors = self._palette()
        active = self.activeAction()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        update_action = getattr(self._app, "update_action", None)
        update_available = getattr(self._app, "_update_available", False)
        for action in self.actions():
            letter = action.data()
            if action.isSeparator() or not isinstance(letter, str) or not letter:
                continue
            self._paint_badge(painter, self.actionGeometry(action), letter,
                              action.isEnabled(), action is active, colors)
            # Green "update available" dot, raised like a superscript just after the label text
            # (kept off the icon, which would otherwise overlap it).
            if update_available and action is update_action:
                self._paint_update_dot(painter, self.actionGeometry(action), action.text())
        painter.end()

    def _paint_badge(self, painter: QPainter, rect: QRect, letter: str,
                     enabled: bool, selected: bool, colors: dict) -> None:
        x = rect.right() - _BADGE_MARGIN - _BADGE_BOX
        y = rect.center().y() - _BADGE_BOX // 2
        box = QRectF(x, y, _BADGE_BOX, _BADGE_BOX)

        if selected and enabled:
            border = QColor(colors["on_accent"])
            border.setAlpha(150)
            fill = QColor(colors["on_accent"])
            fill.setAlpha(30)
            fg = QColor(colors["on_accent"])
        else:
            border = QColor(colors["border"])
            fill = QColor(colors["panel2"])
            fg = QColor(colors["muted"])
        if not enabled:
            border.setAlpha(70)
            fg.setAlpha(110)

        painter.setPen(QPen(border, 1))
        painter.setBrush(fill)
        painter.drawRoundedRect(box, 4.0, 4.0)
        font = painter.font()
        font.setBold(True)
        font.setPointSizeF(max(7.5, font.pointSizeF() * 0.85))
        painter.setFont(font)
        painter.setPen(fg)
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, letter.upper())

    def _paint_update_dot(self, painter: QPainter, rect: QRect, text: str) -> None:
        """Paint a small raised green dot just after the item's text (superscript-style)."""
        fm = self.fontMetrics()
        # Text starts after the QSS item left-padding (15px) + the reserved icon column.
        icon_extent = self.style().pixelMetric(QStyle.PixelMetric.PM_SmallIconSize)
        text_left = rect.left() + 15 + icon_extent + 5
        cx = text_left + fm.horizontalAdvance(text) + 8
        cy = rect.center().y() - max(3, fm.ascent() // 3)  # lifted above the baseline
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(46, 204, 113))  # green
        painter.drawEllipse(QPointF(float(cx), float(cy)), 3.5, 3.5)


class TrayMixin:
    """System-tray icon, context menu and recording-state icon."""

    def _build_recording_tray_icon(self, level: float) -> QIcon:
        """Render a minimal tray icon with live audio bars for the recording state."""
        if is_MACOS:
            macos_icon = self._build_macos_recording_tray_icon(level)
            if not macos_icon.isNull():
                return macos_icon

        level = max(0.0, min(1.0, level))
        active_bars = max(1, min(5, int(round(level * 5))))
        hot = level >= 0.85
        # The rendered icon only has ~10 discrete looks (5 bar counts × hot/cool). The
        # 80 ms refresh timer would otherwise allocate a fresh QPixmap+QPainter+QIcon
        # ~12×/s; cache the finished QIcon per visual state instead.
        cache_key = (active_bars, hot)
        cached = self._recording_icon_cache.get(cache_key)
        if cached is not None:
            return cached

        pixmap = QPixmap(16, 16)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        painter.setPen(QPen(QColor(65, 65, 65)))
        painter.setBrush(QColor(34, 34, 34))
        painter.drawRect(0, 0, 15, 15)

        bar_heights = [4, 7, 10, 7, 4]
        active_color = QColor(255, 120, 70) if hot else QColor(50, 205, 120)
        inactive_color = QColor(80, 80, 80)

        for index, bar_height in enumerate(bar_heights):
            x = 2 + index * 2
            y = 13 - bar_height
            color = active_color if index < active_bars else inactive_color
            painter.fillRect(x, y, 1, bar_height, color)

        painter.end()
        icon = QIcon(pixmap)
        self._recording_icon_cache[cache_key] = icon
        return icon

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
        if is_MACOS and self.macos_audio_recorder:
            try:
                self.macos_audio_recorder.updateMeters()
                average_power = float(self.macos_audio_recorder.averagePowerForChannel_(0))
                self.latest_audio_level = max(0.0, min(1.0, (average_power + 60.0) / 60.0))
            except Exception:
                pass
        self.tray_icon.setIcon(self._build_recording_tray_icon(self.latest_audio_level))

    def show_tray_balloon(self, message: str, timeout_ms: int = 2000, spinner: bool = False,
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
            self._idle_tray_icon = self._get_macos_tray_icon()
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
        tray_menu = _BadgeTrayMenu(self)
        self.tray_menu = tray_menu

        _sp = QStyle.StandardPixmap

        def _ico(pix: QStyle.StandardPixmap) -> QIcon:
            return self.style().standardIcon(pix)

        def _add(pix: QStyle.StandardPixmap, label_key: str, key: str) -> QAction:
            # _BadgeTrayMenu paints <key> as a rounded badge on the far right and triggers the
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
        self._git_root = find_git_root()
        self.update_action = None
        if self._git_root:
            self.update_action = _add(_sp.SP_ArrowDown, "tray_update_action", "u")
            self.update_action.triggered.connect(self.check_for_updates)
            # A rebuild (e.g. language change) must not lose an already-detected update — re-apply
            # the green-dot indicator from the persisted flag.
            self._apply_update_indicator()

        restart_action = _add(_sp.SP_BrowserReload, "tray_restart_action", "n")
        restart_action.triggered.connect(self.restart_app)

        quit_action = _add(_sp.SP_DialogCloseButton, "tray_quit_action", "q")
        quit_action.triggered.connect(self.quit_app)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

        # Connect activation signal for double-click handling
        self.tray_icon.activated.connect(self.on_tray_icon_activated)

        # Initial state update for log file action
        self.update_logfile_menu_action()
        self.update_play_last_recording_action()

        # Kick off the periodic background update check (source checkouts only).
        self._start_update_watcher()

    def on_tray_icon_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        """
        Handles activation events for the system tray icon, like double-clicks.

        Args:
            reason (QSystemTrayIcon.ActivationReason): The reason for the activation.
        """
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            if self.config.get("systray_double_click_copy", True):
                logging.debug("Tray icon double-clicked, copying last transcription.")
                self.copy_last_transcription_to_clipboard()

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
        if not self._has_valid_api_settings():
            self.show_tray_balloon(self.translator.tr("recording_no_api_keys"), 2500)
            self.show_settings_window()
            return

        latest = self._get_latest_recording_path()
        if not latest or not os.path.isfile(latest):
            self.show_tray_balloon(self.translator.tr("no_recording_found_message"), 2000)
            self.update_play_last_recording_action()
            return

        logging.info(f"Re-transcribing last recording: {latest}")
        # No fresh selection context on a manual re-transcribe of an existing file.
        # Result goes to the clipboard (the user hasn't focused a text field).
        self.current_transcription_context = ""
        self.start_transcription_worker(latest, output_mode="clipboard")

    def transcribe_audio_file(self) -> None:
        """Pick an audio (or, with ffmpeg, video) file and transcribe it; result goes to the clipboard."""
        if not self._has_valid_api_settings():
            self.show_tray_balloon(self.translator.tr("recording_no_api_keys"), 2500)
            self.show_settings_window()
            return

        # With ffmpeg available we can also accept video containers (audio is extracted first).
        ffmpeg_exe = resolve_ffmpeg(self.config.get("ffmpeg_path", ""))
        audio_globs = "*.mp3 *.ogg *.wav *.m4a *.flac *.webm *.mp4 *.mpga *.mpeg"
        audio_filter = f"Audio ({audio_globs})"
        if ffmpeg_exe:
            video_globs = " ".join(f"*{ext}" for ext in sorted(VIDEO_EXTENSIONS))
            media_filter = f"{self.translator.tr('media_filter_label')} ({audio_globs} {video_globs})"
            file_filter = f"{media_filter};;{audio_filter};;All files (*)"
        else:
            file_filter = f"{audio_filter};;All files (*)"

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
            self.show_tray_balloon(self.translator.tr("video_needs_ffmpeg_message"), 4000)
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
        from PyQt6.QtCore import QProcess

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
            self.show_tray_balloon(self.translator.tr("restart_failed_message"), 2500)
            return

        app = QApplication.instance()
        if app is not None:
            app.quit()

    # --- Self-update (git pull) ---------------------------------------------------------
    def check_for_updates(self) -> None:
        """Ask for confirmation, then pull the latest code from git for this source checkout.

        Only wired up when running from a git working tree (see ``init_tray_icon``). The pull
        runs asynchronously so the UI never freezes on the network round-trip; when it finishes
        the result is shown and — on success — the user is offered an immediate restart.
        """
        from PyQt6.QtWidgets import QMessageBox

        root = getattr(self, "_git_root", None) or find_git_root()
        if not root:
            return

        # A previous pull is still running — don't start a second one.
        if getattr(self, "_update_process", None) is not None:
            self.show_tray_balloon(self.translator.tr("update_running_message"), 2000)
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

        self._start_git_pull(root)

    def _start_git_pull(self, root: str) -> None:
        """Launch ``git pull --ff-only`` in ``root`` via QProcess (non-blocking)."""
        from PyQt6.QtCore import QProcess

        logging.info("Running git update in %s", root)
        self.show_tray_balloon(self.translator.tr("update_running_message"), 0, spinner=True)

        # Remember HEAD so the finished handler can tell a real update from "already up to date"
        # without depending on git's (localized) wording.
        self._pre_update_head = current_head(root)

        process = QProcess(self)
        self._update_process = process
        process.setWorkingDirectory(root)
        # Fold stderr into stdout so the single readAll() captures the full git report.
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.finished.connect(self._on_git_update_finished)
        process.errorOccurred.connect(self._on_git_update_error)
        # --ff-only keeps this safe as a one-click action: if the local branch has diverged
        # (unpushed commits, dirty tree) git refuses with a clear message instead of opening
        # a blocking merge-commit editor.
        process.start("git", ["pull", "--ff-only"])

    def _on_git_update_error(self, error: object) -> None:
        """Handle a git process that failed to even start (e.g. removed from PATH mid-session)."""
        from PyQt6.QtWidgets import QMessageBox

        if getattr(self, "_update_process", None) is None:
            return  # already handled by finished()
        self._update_process = None
        self.hide_tray_balloon()
        logging.error("git update process error: %s", error)
        QMessageBox.warning(self, self.translator.tr("update_dialog_title"),
                            self.translator.tr("update_git_missing_text"))

    def _on_git_update_finished(self, exit_code: int, exit_status: object) -> None:
        """Report the git pull result and, on success, offer to restart the app."""
        from PyQt6.QtWidgets import QMessageBox

        process = getattr(self, "_update_process", None)
        if process is None:
            return
        self._update_process = None
        self.hide_tray_balloon()

        try:
            output = bytes(process.readAll()).decode("utf-8", errors="replace").strip()
        except Exception:
            output = ""
        process.deleteLater()
        logging.info("git update finished (exit=%s): %s", exit_code, output)

        if exit_code != 0:
            QMessageBox.warning(
                self,
                self.translator.tr("update_dialog_title"),
                self.translator.tr("update_failed_text",
                                    output=output or self.translator.tr("update_no_output")),
            )
            return

        pre_head = getattr(self, "_pre_update_head", None)
        self._pre_update_head = None
        new_head = current_head(getattr(self, "_git_root", None) or "")
        # Nothing changed (already up to date) — no restart is needed, so just inform the user
        # instead of prompting to reboot.
        if pre_head and new_head and pre_head == new_head:
            QMessageBox.information(
                self,
                self.translator.tr("update_dialog_title"),
                self.translator.tr("update_up_to_date_text"),
            )
            return

        # HEAD moved — the pending update has been consumed. Clear the indicator and re-arm the
        # watcher so a *future* upstream commit is picked up again (relevant if the user declines
        # the restart below and keeps this instance running).
        self._set_update_available(False)
        self._start_update_watcher()

        reply = QMessageBox.question(
            self,
            self.translator.tr("update_dialog_title"),
            self.translator.tr("update_success_text",
                               output=output or self.translator.tr("update_no_output")),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.restart_app()

    # --- Background update watcher ------------------------------------------------------
    def _start_update_watcher(self) -> None:
        """Arm the periodic background check that flags when the upstream branch is ahead.

        No-op unless this is a git checkout with git available, or once an update is already
        known (the indicator is shown, so there's nothing more to poll for).
        """
        if not getattr(self, "_git_root", None) or not git_available():
            return
        if getattr(self, "_update_available", False):
            return

        timer = getattr(self, "_update_watch_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setInterval(_UPDATE_CHECK_INTERVAL_MS)
            timer.timeout.connect(self._check_for_updates_background)
            self._update_watch_timer = timer
        if not timer.isActive():
            timer.start()
        # First check runs soon after launch (not on startup) without waiting a full interval.
        # Schedule it only once — this method re-runs on every tray rebuild (e.g. language
        # change), and each singleShot would otherwise stack another pending fetch.
        if not getattr(self, "_update_initial_check_scheduled", False):
            self._update_initial_check_scheduled = True
            QTimer.singleShot(_UPDATE_CHECK_INITIAL_DELAY_MS, self._check_for_updates_background)

    def _check_for_updates_background(self) -> None:
        """Fetch in the background and, if the upstream branch is ahead, raise the indicator.

        Runs ``git fetch`` via QProcess so the network round-trip never blocks the UI. A failed
        fetch (offline, VPN down, …) is ignored — we simply try again on the next tick.
        """
        from PyQt6.QtCore import QProcess

        root = getattr(self, "_git_root", None)
        if not root or getattr(self, "_update_available", False):
            return
        # Don't overlap with a manual pull or a still-running background fetch.
        if getattr(self, "_update_process", None) is not None:
            return
        if getattr(self, "_update_fetch_process", None) is not None:
            return

        process = QProcess(self)
        self._update_fetch_process = process
        process.setWorkingDirectory(root)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.finished.connect(self._on_update_fetch_finished)
        # errorOccurred (e.g. git vanished) just clears the ref; treated like any offline failure.
        process.errorOccurred.connect(lambda _err: setattr(self, "_update_fetch_process", None))
        process.start("git", ["fetch", "--quiet"])

    def _on_update_fetch_finished(self, exit_code: int, exit_status: object) -> None:
        """After a background fetch, compare against upstream and flag an available update."""
        process = getattr(self, "_update_fetch_process", None)
        self._update_fetch_process = None
        if process is not None:
            process.deleteLater()

        # Non-zero exit == offline / fetch failed → ignore, keep polling on the next tick.
        if exit_code != 0:
            logging.debug("Background update fetch failed (exit=%s) — ignored.", exit_code)
            return

        behind = count_behind_upstream(getattr(self, "_git_root", None) or "")
        logging.debug("Background update check: %s commit(s) behind upstream.", behind)
        if behind and behind > 0:
            self._set_update_available(True)

    def _set_update_available(self, available: bool) -> None:
        """Record update availability, update the menu indicator, and (un)arm the watcher."""
        self._update_available = available
        self._apply_update_indicator()
        if available:
            # Nothing more to poll for — the user can see it in the menu now.
            timer = getattr(self, "_update_watch_timer", None)
            if timer is not None and timer.isActive():
                timer.stop()

    def _apply_update_indicator(self) -> None:
        """Reflect the current ``_update_available`` flag on the tray menu entry.

        The green dot itself is painted by ``_BadgeTrayMenu`` after the label text; here we only
        swap the label and trigger a repaint so it appears/disappears.
        """
        action = getattr(self, "update_action", None)
        if action is None:
            return
        available = getattr(self, "_update_available", False)
        label_key = "tray_update_available_action" if available else "tray_update_action"
        action.setText(self.translator.tr(label_key))
        menu = getattr(self, "tray_menu", None)
        if menu is not None:
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

    def _get_app_icon(self) -> QIcon:
        """Return the platform-appropriate application icon."""
        icon_candidates: List[str] = []
        if is_MACOS:
            icon_candidates.append(resource_path("resources", "whispertyper.icns"))
        icon_candidates.append(resource_path("resources", "app_icon.png"))

        for icon_path in icon_candidates:
            if os.path.exists(icon_path):
                icon = QIcon(icon_path)
                if not icon.isNull():
                    return icon
        return QIcon()
