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


def test_fetch_ref_survives_a_crash_so_finished_can_clean_up(monkeypatch, tmp_path):
    updater = make_updater(monkeypatch, tmp_path)
    updater._fetch = updater._process(updater._on_fetch_finished)
    updater._on_fetch_error(QProcess.ProcessError.Crashed)
    assert updater._fetch is not None
    updater._on_fetch_finished(1, QProcess.ExitStatus.CrashExit)
    assert updater._fetch is None and not updater.available

    updater._fetch = updater._process(updater._on_fetch_finished)
    updater._on_fetch_error(QProcess.ProcessError.FailedToStart)
    assert updater._fetch is None
