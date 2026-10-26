"""Install Python packages into WhisperTyper's environment, showing the installer's output live.

Used by the local engine (its optional packages), after a self-update that changed the
requirements, and at a start that finds a package missing. Imports nothing beyond PyQt6, so the
start can show it while other packages are missing.
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional

from PyQt6.QtCore import QProcess, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.platform import installation
from app.ui import message_dialog

#: Looks up a translated text: ``tr(key, **values)``.
Translate = Callable[..., str]


class PackageInstallDialog(QDialog):
    """Runs an install command; ``succeeded`` tells the outcome once ``exec()`` returns."""

    def __init__(self, parent: Optional[QWidget], tr: Translate, title: str, command: List[str]) -> None:
        """Start the command at once; the dialog shows its output until it ends."""
        super().__init__(parent)
        self._tr = tr
        self.setWindowTitle(title)
        self.resize(720, 420)
        layout = QVBoxLayout(self)
        self._label = QLabel(tr("package_install_running"), self)
        self._label.setWordWrap(True)
        layout.addWidget(self._label)
        self._progress = QProgressBar(self)
        self._progress.setRange(0, 0)  # busy: pip reports no overall progress
        layout.addWidget(self._progress)
        self._output = QPlainTextEdit(self)
        self._output.setReadOnly(True)
        self._output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self._output, 1)
        self._buttons = QDialogButtonBox(self)
        self._cancel = QPushButton(tr("package_install_cancel"), self)
        self._buttons.addButton(self._cancel, QDialogButtonBox.ButtonRole.RejectRole)
        self._cancel.clicked.connect(self._on_cancel)
        layout.addWidget(self._buttons)
        self.succeeded = False

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read_output)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_error)
        display = installation.display_command(command)
        logging.info("package_install start: %s", display)
        self._output.appendPlainText(f"$ {display}\n")
        self._process.start(command[0], command[1:])

    def _read_output(self) -> None:
        text = bytes(self._process.readAllStandardOutput().data()).decode("utf-8", "replace")
        for line in text.splitlines():
            if line.strip():
                logging.info("pip: %s", line.rstrip())
        self._output.moveCursor(self._output.textCursor().MoveOperation.End)
        self._output.insertPlainText(text)
        self._output.ensureCursorVisible()

    def _on_finished(self, code: int, _status: QProcess.ExitStatus) -> None:
        self._read_output()
        self.succeeded = code == 0
        logging.info("package_install finished: exit code %s", code)
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        self._label.setText(self._tr("package_install_done" if self.succeeded else "package_install_failed",
                                     code=code))
        self._cancel.setText(self._tr("package_install_close"))
        self._cancel.clicked.disconnect()
        self._cancel.clicked.connect(self.accept if self.succeeded else self.reject)

    def _on_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            logging.error("package_install could not start the installer: %s", self._process.errorString())
            self._output.appendPlainText(self._process.errorString())
            self._on_finished(-1, QProcess.ExitStatus.CrashExit)

    def _on_cancel(self) -> None:
        if self._process.state() != QProcess.ProcessState.NotRunning:
            logging.info("package_install cancelled by the user")
            self._process.kill()
            self._process.waitForFinished(3000)
        self.reject()

    def keyPressEvent(self, event: Optional[QKeyEvent]) -> None:  # noqa: N802 - Qt API
        """Escape cancels like the button (stopping the installer first)."""
        if event is not None and event.key() == Qt.Key.Key_Escape:
            self._on_cancel()
            return
        super().keyPressEvent(event)


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
