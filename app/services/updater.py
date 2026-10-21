"""Self-update for source checkouts: background upstream watcher, ``git pull --ff-only`` and a forced update.

Git runs through QProcess so neither the periodic fetch nor the pull ever blocks the GUI.
The UI layer decides what to show; this class only reports outcomes through signals.
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional

from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, pyqtSignal

from app.services.gitutil import changed_paths, count_behind_upstream, current_head, find_git_root, git_available

# The first fetch runs shortly after launch so startup isn't slowed; after that the watcher
# polls until an update is found. Offline / failed fetches are retried on the next tick.
UPDATE_CHECK_INITIAL_DELAY_MS = 20_000       # ~20 s after launch
UPDATE_CHECK_INTERVAL_MS = 30 * 60 * 1000    # then every 30 minutes
# A fetch/pull that hangs (stalled network, blocked credential helper) is killed, so the next
# check can run and the tray spinner never stays up forever.
GIT_TIMEOUT_MS = 120_000
#: Message of the stash a forced update keeps the local changes in.
FORCE_UPDATE_STASH_MESSAGE = "WhisperTyper: local changes before a forced update"


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
        # The git commands of the running update that are still to come, and their collected output.
        self._pending_steps: List[List[str]] = []
        self._pull_output: List[str] = []
        #: The last update changed requirements.txt (dependencies to install before restarting).
        self.requirements_changed = False
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
        # --ff-only keeps this safe as a one-click action: if the local branch has diverged
        # (unpushed commits, dirty tree) git refuses with a clear message instead of opening
        # a blocking merge-commit editor.
        self._run_update([["pull", "--ff-only"]])

    def force_update(self) -> None:
        """Move the checkout to the upstream state even with local changes or commits.

        Local changes (untracked files included, ignored ones such as the config untouched) are
        stashed first, so ``git stash list`` still has them; replaced local commits stay in the
        reflog, and the output names the previous HEAD. The result arrives via ``pull_finished``.
        """
        # Without an upstream (detached HEAD, untracked branch) the reset would fail after the
        # stash had already taken the local changes away, so that is checked first.
        self._run_update([["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
                          ["fetch", "--quiet"],
                          ["stash", "push", "--include-untracked", "-m", FORCE_UPDATE_STASH_MESSAGE],
                          ["reset", "--hard", "@{u}"]])

    def _run_update(self, steps: List[List[str]]) -> None:
        """Run the git commands one after another; the first failing one ends the update."""
        if not self.root or self._pull is not None:
            return
        if self._fetch is not None:
            # Two git processes would fight over the ref locks; the update fetches anyway.
            self._fetch.kill()
            self._fetch.waitForFinished(2000)
        self.requirements_changed = False
        logging.info("Running git update %s in %s", " / ".join(" ".join(step) for step in steps), self.root)
        # HEAD before the pull tells a real update from "already up to date" without
        # depending on git's localized wording.
        self._pre_pull_head = current_head(self.root)
        self._pending_steps = list(steps)
        self._pull_output = []
        if len(steps) > 1 and self._pre_pull_head:
            self._pull_output.append(f"HEAD {self._pre_pull_head}")
        self._start_next_step()

    def _start_next_step(self) -> None:
        process = self._process(self._on_pull_finished)
        process.errorOccurred.connect(self._on_pull_error)
        self._pull = process
        process.start("git", self._pending_steps.pop(0))

    def _process(self, on_finished: Callable[[int, QProcess.ExitStatus], None]) -> QProcess:
        """A git QProcess in the checkout with stderr folded into stdout, killed after ``GIT_TIMEOUT_MS``.

        Git must never wait for a password on the hidden terminal.
        """
        process = QProcess(self)
        process.setWorkingDirectory(self.root or "")
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("GIT_TERMINAL_PROMPT", "0")
        process.setProcessEnvironment(environment)
        process.finished.connect(on_finished)
        watchdog = QTimer(process)
        watchdog.setSingleShot(True)
        watchdog.timeout.connect(process.kill)
        watchdog.start(GIT_TIMEOUT_MS)
        return process

    def _set_available(self, available: bool) -> None:
        """Record availability; once an update is known there is nothing more to poll for."""
        self.available = available
        if available:
            self._timer.stop()
        self.availability_changed.emit(available)

    def _on_pull_error(self, error: QProcess.ProcessError) -> None:
        """Git failed to start; ``finished`` will not follow (it does for a crash or a kill)."""
        if error != QProcess.ProcessError.FailedToStart or self._pull is None:
            return  # handled by finished()
        process, self._pull = self._pull, None
        process.deleteLater()
        self._pending_steps = []
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
        logging.info("git update step finished (exit=%s): %s", exit_code, output)
        if output:
            self._pull_output.append(output)
        if exit_code == 0 and self._pending_steps:
            self._start_next_step()
            return
        self._pending_steps = []
        output = "\n".join(self._pull_output)
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
            # New requirements must be installed before the restart, or the new code cannot start.
            self.requirements_changed = bool(pre_head and new_head and changed_paths(
                self.root or "", pre_head, new_head, ["requirements.txt"]))
        self.pull_finished.emit(True, changed, output)

    def _fetch_in_background(self) -> None:
        """``git fetch`` without blocking; a failed fetch (offline, VPN down, …) is ignored."""
        if not self.root or self.available or self._pull is not None or self._fetch is not None:
            return
        process = self._process(self._on_fetch_finished)
        process.errorOccurred.connect(self._on_fetch_error)
        self._fetch = process
        process.start("git", ["fetch", "--quiet"])

    def _on_fetch_error(self, error: QProcess.ProcessError) -> None:
        """Git failed to start (e.g. vanished): clear the ref like any offline failure."""
        if error != QProcess.ProcessError.FailedToStart or self._fetch is None:
            return  # handled by finished()
        process, self._fetch = self._fetch, None
        process.deleteLater()

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
