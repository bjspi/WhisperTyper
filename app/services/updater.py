"""Self-update for source checkouts: background upstream watcher and ``git pull --ff-only``.

Git runs through QProcess so neither the periodic fetch nor the pull ever blocks the GUI.
The UI layer decides what to show; this class only reports outcomes through signals.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from PyQt6.QtCore import QObject, QProcess, QTimer, pyqtSignal

from app.services.gitutil import count_behind_upstream, current_head, find_git_root, git_available

# The first fetch runs shortly after launch so startup isn't slowed; after that the watcher
# polls until an update is found. Offline / failed fetches are retried on the next tick.
UPDATE_CHECK_INITIAL_DELAY_MS = 20_000       # ~20 s after launch
UPDATE_CHECK_INTERVAL_MS = 30 * 60 * 1000    # then every 30 minutes


class GitUpdater(QObject):
    """Track whether upstream is ahead and pull it on request."""

    #: The upstream branch is ahead (True) or the pending update was consumed (False).
    availability_changed = pyqtSignal(bool)
    #: A pull finished: (succeeded, HEAD moved, git output).
    pull_finished = pyqtSignal(bool, bool, str)
    #: git could not be started for the pull (e.g. removed from PATH mid-session).
    pull_failed_to_start = pyqtSignal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        """Locate the checkout once; ``root`` stays None for frozen builds."""
        super().__init__(parent)
        self.root: Optional[str] = find_git_root()
        self.available = False
        self._pull: Optional[QProcess] = None
        self._fetch: Optional[QProcess] = None
        self._pre_pull_head: Optional[str] = None
        self._initial_check_scheduled = False
        self._timer = QTimer(self)
        self._timer.setInterval(UPDATE_CHECK_INTERVAL_MS)
        self._timer.timeout.connect(self._fetch_in_background)

    @property
    def pulling(self) -> bool:
        """Whether a pull is in progress."""
        return self._pull is not None

    def start_watching(self) -> None:
        """Arm the periodic upstream check (no-op without a checkout/git or once an update is known)."""
        if not self.root or not git_available() or self.available:
            return
        if not self._timer.isActive():
            self._timer.start()
        # The first check runs soon after launch without waiting a full interval — scheduled
        # once, however often the tray is rebuilt (e.g. on a language change).
        if not self._initial_check_scheduled:
            self._initial_check_scheduled = True
            QTimer.singleShot(UPDATE_CHECK_INITIAL_DELAY_MS, self._fetch_in_background)

    def pull(self) -> None:
        """Run ``git pull --ff-only`` in the checkout; the result arrives via ``pull_finished``."""
        if not self.root or self._pull is not None:
            return
        logging.info("Running git update in %s", self.root)
        # HEAD before the pull tells a real update from "already up to date" without
        # depending on git's localized wording.
        self._pre_pull_head = current_head(self.root)
        process = self._process(self._on_pull_finished)
        process.errorOccurred.connect(self._on_pull_error)
        self._pull = process
        # --ff-only keeps this safe as a one-click action: if the local branch has diverged
        # (unpushed commits, dirty tree) git refuses with a clear message instead of opening
        # a blocking merge-commit editor.
        process.start("git", ["pull", "--ff-only"])

    def _process(self, on_finished: Callable[[int, QProcess.ExitStatus], None]) -> QProcess:
        """A git QProcess in the checkout with stderr folded into stdout."""
        process = QProcess(self)
        process.setWorkingDirectory(self.root or "")
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.finished.connect(on_finished)
        return process

    def _set_available(self, available: bool) -> None:
        """Record availability; once an update is known there is nothing more to poll for."""
        self.available = available
        if available:
            self._timer.stop()
        self.availability_changed.emit(available)

    def _on_pull_error(self, error: object) -> None:
        """Git failed to start; ``finished`` will not follow."""
        if self._pull is None:
            return  # already handled by finished()
        self._pull = None
        logging.error("git update process error: %s", error)
        self.pull_failed_to_start.emit()

    def _on_pull_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        """Report the pull and re-arm the watcher when HEAD moved."""
        process, self._pull = self._pull, None
        if process is None:
            return
        try:
            output = bytes(process.readAll().data()).decode("utf-8", errors="replace").strip()
        except Exception:
            output = ""
        process.deleteLater()
        logging.info("git update finished (exit=%s): %s", exit_code, output)
        pre_head, self._pre_pull_head = self._pre_pull_head, None
        if exit_code != 0:
            self.pull_finished.emit(False, False, output)
            return
        new_head = current_head(self.root or "")
        changed = not (pre_head and new_head and pre_head == new_head)
        if changed:
            # The pending update was consumed; watch for the next upstream commit.
            self._set_available(False)
            self.start_watching()
        self.pull_finished.emit(True, changed, output)

    def _fetch_in_background(self) -> None:
        """``git fetch`` without blocking; a failed fetch (offline, VPN down, …) is ignored."""
        if not self.root or self.available or self._pull is not None or self._fetch is not None:
            return
        process = self._process(self._on_fetch_finished)
        # errorOccurred (e.g. git vanished) just clears the ref, like any offline failure.
        process.errorOccurred.connect(lambda _error: setattr(self, "_fetch", None))
        self._fetch = process
        process.start("git", ["fetch", "--quiet"])

    def _on_fetch_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        """Compare against upstream after a successful fetch."""
        process, self._fetch = self._fetch, None
        if process is not None:
            process.deleteLater()
        if exit_code != 0:
            logging.debug("Background update fetch failed (exit=%s) — ignored.", exit_code)
            return
        behind = count_behind_upstream(self.root or "")
        logging.debug("Background update check: %s commit(s) behind upstream.", behind)
        if behind and behind > 0:
            self._set_available(True)
