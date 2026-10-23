"""Git self-update processes: no terminal prompts, a watchdog, and only a missing git ends a pull early."""
from __future__ import annotations

import os
import sys
import time

import pytest

pytest.importorskip("PyQt6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QProcess  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from app.services import updater as updater_module  # noqa: E402
from app.services.releases import Release  # noqa: E402
from app.services.updater import GitUpdater  # noqa: E402

QAPP = QApplication.instance() or QApplication([])


def make_updater(monkeypatch, tmp_path) -> GitUpdater:
    monkeypatch.setattr(updater_module, "find_git_root", lambda: str(tmp_path))
    return GitUpdater()


def test_git_processes_never_prompt_and_are_killed_when_hung(monkeypatch, tmp_path):
    monkeypatch.setattr(updater_module, "GIT_TIMEOUT_MS", 300)
    updater = make_updater(monkeypatch, tmp_path)
    outcomes = []
    process = updater._process(lambda code, status: outcomes.append((code, status)))
    assert process.processEnvironment().value("GIT_TERMINAL_PROMPT") == "0"

    process.start(sys.executable, ["-c", "import time; time.sleep(30)"])
    assert process.waitForStarted(5000)
    deadline = time.monotonic() + 10
    while not outcomes and time.monotonic() < deadline:
        QAPP.processEvents()
        time.sleep(0.01)
    assert outcomes and outcomes[0][1] == QProcess.ExitStatus.CrashExit
    process.deleteLater()


def test_only_a_missing_git_counts_as_failed_to_start(monkeypatch, tmp_path):
    updater = make_updater(monkeypatch, tmp_path)
    failed_to_start = []
    updater.pull_failed_to_start.connect(lambda: failed_to_start.append(True))
    updater._pull = updater._process(updater._on_pull_finished)

    # A crash (or the watchdog's kill) is followed by ``finished``, which reports the outcome.
    updater._on_pull_error(QProcess.ProcessError.Crashed)
    assert updater._pull is not None and failed_to_start == []

    updater._on_pull_error(QProcess.ProcessError.FailedToStart)
    assert updater._pull is None and failed_to_start == [True]
    updater._on_pull_error(QProcess.ProcessError.FailedToStart)  # No pull: nothing to report.
    assert failed_to_start == [True]


def test_check_ref_survives_a_crash_so_finished_can_clean_up(monkeypatch, tmp_path):
    updater = make_updater(monkeypatch, tmp_path)
    updater._check = updater._process(updater._on_check_finished)
    updater._on_check_error(QProcess.ProcessError.Crashed)
    assert updater._check is not None
    updater._on_check_finished(1, QProcess.ExitStatus.CrashExit)
    assert updater._check is None and not updater.available

    updater._check = updater._process(updater._on_check_finished)
    updater._on_check_error(QProcess.ProcessError.FailedToStart)
    assert updater._check is None and not updater.checking


class _FakeProcess:
    """Stands in for a finished ``git ls-remote`` QProcess."""

    def __init__(self, output: str) -> None:
        self._output = output.encode()

    def readAll(self):
        return type("Bytes", (), {"data": lambda _self: self._output})()

    def deleteLater(self) -> None:
        pass


def _ls_remote(*tags: str) -> str:
    return "".join(f"{index:040x}\trefs/tags/{tag}\n" for index, tag in enumerate(tags))


def test_only_a_newer_published_release_is_offered(monkeypatch, tmp_path):
    updater = make_updater(monkeypatch, tmp_path)
    updater.current = Release(0, 15, 1)
    asked = []
    updater.ask_published = lambda release, answer: asked.append((release, answer))
    started = []
    monkeypatch.setattr(QProcess, "start", lambda self, program, arguments: started.append(arguments))
    answers = []
    updater.checked.connect(answers.append)

    # Commits on main or an older tag are no update.
    updater._check = _FakeProcess(_ls_remote("v0.15.0", "v0.15.1", "v0.16.0-beta"))
    updater._on_check_finished(0, QProcess.ExitStatus.NormalExit)
    assert answers == [True] and asked == [] and not updater.available

    # A tag whose release is not published yet waits for the retry.
    updater._check = _FakeProcess(_ls_remote("v0.15.1", "v0.15.2"))
    updater._on_check_finished(0, QProcess.ExitStatus.NormalExit)
    release, answer = asked.pop()
    assert release == Release(0, 15, 2) and updater.checking
    answer(False)
    assert not updater.checking and not updater.available and updater._unpublished_retry.isActive()

    # A published release is offered once its tag was fetched for the change list.
    updater._check = _FakeProcess(_ls_remote("v0.15.2"))
    updater._on_check_finished(0, QProcess.ExitStatus.NormalExit)
    asked.pop()[1](True)
    assert started[-1][-1] == "+refs/tags/v0.15.2:refs/tags/v0.15.2"
    updater._check = _FakeProcess("")
    updater._on_tag_fetched(Release(0, 15, 2), 0)
    assert updater.latest == Release(0, 15, 2) and not updater.checking
    updater._unpublished_retry.stop()


def test_updates_move_the_checkout_to_the_release_tag(monkeypatch, tmp_path):
    updater = make_updater(monkeypatch, tmp_path)
    started = []
    monkeypatch.setattr(updater, "_start_next_step", lambda: started.append(list(updater._pending_steps)))
    updater.latest = Release(0, 16, 0)

    updater.pull()
    assert started[-1][-1] == ["merge", "--ff-only", "refs/tags/v0.16.0"]
    updater.force_update()
    assert started[-1][-2][:2] == ["stash", "push"]
    assert started[-1][-1] == ["reset", "--hard", "refs/tags/v0.16.0"]
    assert updater.target == Release(0, 16, 0)


def test_details_toggle_names_what_a_click_does():
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QPlainTextEdit, QPushButton

    from app.ui.update_dialog import run_update_dialog

    seen = []

    def click_toggle_twice() -> None:
        dialog = next(w for w in QAPP.topLevelWidgets() if isinstance(w, QDialog) and w.isVisible())
        toggle = next(b for b in dialog.findChildren(QPushButton) if b.isCheckable())
        details = dialog.findChild(QPlainTextEdit)
        seen.append((toggle.text(), details.isVisible()))
        toggle.click()
        seen.append((toggle.text(), details.isVisible()))
        toggle.click()
        seen.append((toggle.text(), details.isVisible()))
        dialog.reject()

    QTimer.singleShot(0, click_toggle_twice)
    run_update_dialog(None, "Title", "Message", "diff",
                      [("Close", QDialogButtonBox.ButtonRole.RejectRole)], details_captions=("Show", "Hide"))
    assert seen == [("Show", False), ("Hide", True), ("Show", False)]
