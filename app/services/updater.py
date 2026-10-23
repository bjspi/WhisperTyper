"""Self-update for source checkouts: background release watcher, an update to the newest release and a forced one.

Updates follow release tags (``vX.Y.Z``), never single commits, so pushing to ``main`` does not
announce an update; a new tag counts once its GitHub release is published (see ``app.services.releases``).
Git runs through QProcess so neither the periodic check nor the update ever blocks the GUI.
The UI layer decides what to show; this class only reports outcomes through signals.
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional

from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, QUrl, pyqtSignal
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest

from app import __version__
from app.services.gitutil import changed_paths, current_head, find_git_root, git_available
from app.services.releases import Release, newest_release, remote_tags

# The first check runs shortly after launch so startup isn't slowed; after that the watcher
# polls until an update is found. Offline / failed checks are retried on the next tick.
UPDATE_CHECK_INITIAL_DELAY_MS = 20_000       # ~20 s after launch
UPDATE_CHECK_INTERVAL_MS = 30 * 60 * 1000    # then every 30 minutes
# A tag's release is published once its CI run passed, some minutes after the tag was pushed.
UNPUBLISHED_RETRY_MS = 5 * 60 * 1000
PUBLISHED_CHECK_TIMEOUT_MS = 20_000
# A git call that hangs (stalled network, blocked credential helper) is killed, so the next
# check can run and the tray spinner never stays up forever.
GIT_TIMEOUT_MS = 120_000
#: Message of the stash a forced update keeps the local changes in.
FORCE_UPDATE_STASH_MESSAGE = "WhisperTyper: local changes before a forced update"
_NOT_FOUND = 404

#: Answers whether a release is published (True/False), or None when GitHub could not be asked.
PublishedCheck = Callable[[Release, Callable[[Optional[bool]], None]], None]


class GitUpdater(QObject):
    """Find the newest published release and move the checkout to it on request."""

    #: A newer release was found (True) or the pending update was installed (False).
    availability_changed = pyqtSignal(bool)
    #: A check finished: whether the release list could be loaded.
    checked = pyqtSignal(bool)
    #: An update finished: (succeeded, HEAD moved, git output).
    pull_finished = pyqtSignal(bool, bool, str)
    #: git could not be started for the update (e.g. removed from PATH mid-session).
    pull_failed_to_start = pyqtSignal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        """Locate the checkout once; ``root`` stays None for frozen builds."""
        super().__init__(parent)
        self.root: Optional[str] = find_git_root()
        self.current = Release.from_version(__version__)
        #: The newest published release, once it is newer than the running version.
        self.latest: Optional[Release] = None
        #: The release the running (or last) update moves the checkout to.
        self.target: Optional[Release] = None
        self._pull: Optional[QProcess] = None
        # The git commands of the running update that are still to come, and their collected output.
        self._pending_steps: List[List[str]] = []
        self._pull_output: List[str] = []
        #: The last update changed requirements.txt (dependencies to install before restarting).
        self.requirements_changed = False
        self._check: Optional[QProcess] = None
        # Between finding a new tag and offering it: is it published, then fetching the tag.
        self._confirming = False
        self.ask_published: PublishedCheck = self._ask_published
        self._network: Optional[QNetworkAccessManager] = None
        self._pre_pull_head: Optional[str] = None
        self._initial_check_scheduled = False
        self._timer = QTimer(self)
        self._timer.setInterval(UPDATE_CHECK_INTERVAL_MS)
        self._timer.timeout.connect(self._check_in_background)
        self._unpublished_retry = QTimer(self)
        self._unpublished_retry.setSingleShot(True)
        self._unpublished_retry.setInterval(UNPUBLISHED_RETRY_MS)
        self._unpublished_retry.timeout.connect(self._check_in_background)

    @property
    def available(self) -> bool:
        """Whether a newer release is known."""
        return self.latest is not None

    @property
    def pulling(self) -> bool:
        """Whether an update is in progress."""
        return self._pull is not None

    @property
    def checking(self) -> bool:
        """Whether a release check is in progress."""
        return self._check is not None or self._confirming

    def start_watching(self) -> None:
        """Arm the periodic release check (no-op without a checkout/git or once an update is known)."""
        if not self.root or self.current is None or not git_available() or self.available:
            return
        if not self._timer.isActive():
            self._timer.start()
        # The first check runs soon after launch without waiting a full interval — scheduled
        # once, however often the tray is rebuilt (e.g. on a language change).
        if not self._initial_check_scheduled:
            self._initial_check_scheduled = True
            QTimer.singleShot(UPDATE_CHECK_INITIAL_DELAY_MS, self._check_in_background)

    def check_now(self) -> None:
        """Look for a newer release now; ``checked`` reports when the answer is in."""
        if not self.root or self.current is None or not git_available() or self.pulling:
            self.checked.emit(False)
        elif not self.checking:
            self._start_check()

    def pull(self) -> None:
        """Fast-forward the checkout to the newest release; the result arrives via ``pull_finished``."""
        # --ff-only keeps this safe as a one-click action: if the checkout has diverged (local
        # commits, changes in the way) git refuses with a clear message instead of opening a
        # blocking merge-commit editor. A checkout already past the release stays where it is.
        if self.latest is not None:
            self._run_update(self.latest, [_fetch_tag(self.latest),
                                           ["merge", "--ff-only", f"refs/tags/{self.latest.tag}"]])

    def force_update(self) -> None:
        """Move the checkout to the newest release even with local changes or commits.

        Local changes (untracked files included, ignored ones such as the config untouched) are
        stashed first, so ``git stash list`` still has them; replaced local commits stay in the
        reflog, and the output names the previous HEAD. The result arrives via ``pull_finished``.
        """
        if self.latest is not None:
            self._run_update(self.latest, [_fetch_tag(self.latest),
                                           ["stash", "push", "--include-untracked", "-m", FORCE_UPDATE_STASH_MESSAGE],
                                           ["reset", "--hard", f"refs/tags/{self.latest.tag}"]])

    def _run_update(self, release: Release, steps: List[List[str]]) -> None:
        """Run the git commands one after another; the first failing one ends the update."""
        if not self.root or self._pull is not None:
            return
        if self._check is not None:
            # Two git processes would fight over the ref locks; the update fetches anyway.
            self._check.kill()
            self._check.waitForFinished(2000)
        self.target = release
        self.requirements_changed = False
        logging.info("Updating to %s: %s in %s", release.tag, " / ".join(" ".join(step) for step in steps), self.root)
        # HEAD before the update tells a real update from "already up to date" without
        # depending on git's localized wording.
        self._pre_pull_head = current_head(self.root)
        self._pending_steps = list(steps)
        self._pull_output = []
        if len(steps) > 2 and self._pre_pull_head:
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
        environment.insert("GCM_INTERACTIVE", "Never")
        process.setProcessEnvironment(environment)
        process.finished.connect(on_finished)
        watchdog = QTimer(process)
        watchdog.setSingleShot(True)
        watchdog.timeout.connect(process.kill)
        watchdog.start(GIT_TIMEOUT_MS)
        return process

    def _set_latest(self, release: Optional[Release]) -> None:
        """Record the pending release; once one is known there is nothing more to poll for."""
        self.latest = release
        if release is not None:
            self._timer.stop()
        self.availability_changed.emit(release is not None)

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
        """Report the update and re-arm the watcher when HEAD moved."""
        process, self._pull = self._pull, None
        if process is None:
            return
        output = _read_output(process)
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
            # The pending update was installed; this process still runs the old version, which
            # the restart replaces, so only a release newer than the installed one counts now.
            self.current = self.target
            self._set_latest(None)
            self.start_watching()
            # New requirements must be installed before the restart, or the new code cannot start.
            self.requirements_changed = bool(pre_head and new_head and changed_paths(
                self.root or "", pre_head, new_head, ["requirements.txt"]))
        self.pull_finished.emit(True, changed, output)

    def _check_in_background(self) -> None:
        """A periodic check; a failed one (offline, VPN down, …) is retried on the next tick."""
        if not (self.root is None or self.available or self.pulling or self.checking):
            self._start_check()

    def _start_check(self) -> None:
        """List the remote's tags without touching the checkout."""
        process = self._process(self._on_check_finished)
        process.errorOccurred.connect(self._on_check_error)
        self._check = process
        process.start("git", ["ls-remote", "--tags", "--refs", "origin"])

    def _on_check_error(self, error: QProcess.ProcessError) -> None:
        """Git failed to start (e.g. vanished): clear the ref like any offline failure."""
        if error != QProcess.ProcessError.FailedToStart or self._check is None:
            return  # handled by finished()
        process, self._check = self._check, None
        process.deleteLater()
        self._confirming = False
        self.checked.emit(False)

    def _on_check_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        """Compare the newest release tag with the running version."""
        process, self._check = self._check, None
        output = _read_output(process) if process is not None else ""
        if exit_code != 0:
            logging.debug("Update check failed (exit=%s) and will be retried: %s", exit_code, output)
            self.checked.emit(False)
            return
        newest = newest_release(remote_tags(output))
        logging.debug("Update check: newest release %s, running %s.", newest, __version__)
        if newest is None or self.current is None or newest <= self.current:
            self.checked.emit(True)
            return
        self._confirming = True
        self.ask_published(newest, lambda published: self._on_published(newest, published))

    def _on_published(self, release: Release, published: Optional[bool]) -> None:
        """Fetch a published release's tag, so its changes can be shown before updating.

        An unknown answer (GitHub unreachable while git works, e.g. behind a proxy) does not hold
        the release back: CI passed for it in all likelihood.
        """
        if published is False or self.pulling:
            self._confirming = False
            if published is False:
                logging.info("Release %s is tagged but not published yet.", release.tag)
                self._unpublished_retry.start()
            self.checked.emit(True)
            return
        process = self._process(lambda code, _status: self._on_tag_fetched(release, code))
        process.errorOccurred.connect(self._on_check_error)
        self._check = process
        process.start("git", _fetch_tag(release))

    def _on_tag_fetched(self, release: Release, exit_code: int) -> None:
        process, self._check = self._check, None
        self._confirming = False
        output = _read_output(process) if process is not None else ""
        if exit_code != 0:
            logging.debug("Fetching %s failed (exit=%s): %s", release.tag, exit_code, output)
            self.checked.emit(False)
            return
        self._set_latest(release)
        self.checked.emit(True)

    def _ask_published(self, release: Release, answer: Callable[[Optional[bool]], None]) -> None:
        """Ask GitHub for the release ZIP without following its redirect to the download host."""
        if self._network is None:
            self._network = QNetworkAccessManager(self)
        request = QNetworkRequest(QUrl(release.zip_url))
        request.setAttribute(QNetworkRequest.Attribute.RedirectPolicyAttribute,
                             QNetworkRequest.RedirectPolicy.ManualRedirectPolicy)
        request.setTransferTimeout(PUBLISHED_CHECK_TIMEOUT_MS)
        reply = self._network.head(request)
        if reply is None:
            answer(None)
            return
        reply.finished.connect(lambda: answer(_published(reply)))


def _fetch_tag(release: Release) -> List[str]:
    """Fetch exactly the release's tag (replacing a stale local one), leaving the branches alone."""
    return ["fetch", "--quiet", "--no-tags", "origin", f"+refs/tags/{release.tag}:refs/tags/{release.tag}"]


def _published(reply: QNetworkReply) -> Optional[bool]:
    """True for a found or redirected asset, False for 404, None if GitHub could not be asked."""
    status = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
    reply.deleteLater()
    if not status:
        logging.debug("Release check unanswered: %s", reply.errorString())
        return None
    if status == _NOT_FOUND:
        return False
    return True if 200 <= status < 400 else None


def _read_output(process: QProcess) -> str:
    try:
        output = bytes(process.readAll().data()).decode("utf-8", errors="replace").strip()
    except Exception:
        output = ""
    process.deleteLater()
    return output
