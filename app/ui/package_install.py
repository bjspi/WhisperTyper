"""Install Python packages into WhisperTyper's environment, showing the installer's output live.

Used by the local engine (its optional packages), after a self-update that changed the
requirements, and at a start that finds a package missing. Imports nothing beyond PyQt6, so the
start can show it while other packages are missing.
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional

from PyQt6.QtCore import QProcess
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import QProgressBar, QWidget

from app.platform import installation
from app.ui import message_dialog

#: Looks up a translated text: ``tr(key, **values)``.
Translate = Callable[..., str]


class PackageInstallDialog(message_dialog.MessageDialog):
    """Runs an install command in the shared dialog look; ``succeeded`` tells the outcome after ``exec()``.

    The installer's output fills the details box live. Cancel, Escape and the window's close button
    all stop a running installer before the dialog closes.
    """

    def __init__(self, parent: Optional[QWidget], tr: Translate, title: str, command: List[str]) -> None:
        """Start the command at once; the dialog shows its output until it ends."""
        display = installation.display_command(command)
        super().__init__(parent, title, tr("package_install_running"),
                         [(tr("package_install_cancel"), message_dialog.SECONDARY)],
                         message_dialog.UPDATE, details=f"$ {display}\n", window_title=title)
        self._tr = tr
        self.succeeded = False
        self._progress = QProgressBar(self)
        self._progress.setObjectName("messageProgress")
        self._progress.setTextVisible(False)
        self._progress.setRange(0, 0)  # busy: pip reports no overall progress
        self.body_layout.insertWidget(1, self._progress)
        self._fit(wide=True)  # pip's lines are long

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read_output)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_error)
        logging.info("package_install start: %s", display)
        self._process.start(command[0], command[1:])

    def _read_output(self) -> None:
        text = bytes(self._process.readAllStandardOutput().data()).decode("utf-8", "replace")
        for line in text.splitlines():
            if line.strip():
                logging.info("pip: %s", line.rstrip())
        if self.details is None:
            return
        self.details.moveCursor(QTextCursor.MoveOperation.End)
        self.details.insertPlainText(text)
        self.details.ensureCursorVisible()

    def _on_finished(self, code: int, _status: QProcess.ExitStatus) -> None:
        self._read_output()
        self.succeeded = code == 0
        logging.info("package_install finished: exit code %s", code)
        self._progress.setRange(0, 1)
        self._progress.setValue(1 if self.succeeded else 0)  # a failure must not look like a full bar
        if self.text_label is not None:
            self.text_label.setText(self._tr("package_install_done" if self.succeeded else "package_install_failed",
                                             code=code))
        close = self.buttons[0]
        close.setText(self._tr("package_install_close"))
        close.setObjectName(message_dialog.PRIMARY)  # the finished install's only way on
        style = close.style()
        if style is not None:
            style.unpolish(close)
            style.polish(close)

    def _on_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            logging.error("package_install could not start the installer: %s", self._process.errorString())
            if self.details is not None:
                self.details.appendPlainText(self._process.errorString())
            self._on_finished(-1, QProcess.ExitStatus.CrashExit)

    def _choose(self, caption: str) -> None:
        """The single button cancels while the installer runs and closes once it ended."""
        if self.succeeded:
            self.accept()
        else:
            self.reject()

    def reject(self) -> None:
        """Stop a running installer first; a half-finished install is pip's to roll back."""
        if self._process.state() != QProcess.ProcessState.NotRunning:
            logging.info("package_install cancelled by the user")
            self._process.kill()
            self._process.waitForFinished(3000)
        super().reject()


def run_install(parent: Optional[QWidget], tr: Translate, title: str, command: List[str]) -> bool:
    """Run ``command`` in the install dialog; True when it succeeded."""
    dialog = PackageInstallDialog(parent, tr, title, command)
    dialog.exec()
    return dialog.succeeded


def install_requirements(parent: Optional[QWidget], tr: Translate) -> bool:
    """Install the checkout's requirements; False when that is impossible here or failed."""
    command = installation.requirements_install_command()
    if command is None:
        return False
    return run_install(parent, tr, tr("requirements_install_title"), command)


def offer_missing_package_install(module: str, tr: Translate) -> bool:
    """At a start that misses ``module``: offer the install a checkout can do, else name the repair.

    True when the packages were installed, so the start can be repeated.
    """
    title = tr("missing_package_title")
    text = tr("missing_package_text", module=module)
    if installation.requirements_install_command() is not None:
        install = tr("install_button")
        choice = message_dialog.show_dialog(
            None, title, text + "\n\n" + tr("missing_package_install_text"),
            [(tr("package_install_close"), message_dialog.SECONDARY), (install, message_dialog.PRIMARY)],
            message_dialog.ERROR, window_title="WhisperTyper")
        return choice == install and install_requirements(None, tr)
    command = installation.repair_command()
    advice = (tr("missing_package_command_text", command=command) if command
              else tr("missing_package_reinstall_text"))
    message_dialog.inform(None, title, text + "\n\n" + advice, tr("package_install_close"), message_dialog.ERROR)
    return False
