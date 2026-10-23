"""Install the optional packages of the local engine with pip, showing pip's output live."""
from __future__ import annotations

import logging
import sys
from typing import List, Optional

from PyQt6.QtCore import QProcess, Qt
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

from app.core.i18n import TranslationManager
from app.local_engine.availability import console_python, pip_command
from app.ui import message_dialog


def environment_description() -> str:
    """Where pip installs to: a virtual environment or the Python installation itself."""
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    return f"{'venv' if in_venv else 'Python'}: {sys.prefix}"


def confirm_install(parent: QWidget, translator: TranslationManager, requirements: List[str]) -> bool:
    """Ask before anything is downloaded; names the packages, the size and the target environment."""
    tr = translator.tr
    nvidia = any(req.startswith("nvidia-") for req in requirements)
    text = tr("local_install_confirm_text", packages="\n".join(f"• {req}" for req in requirements),
              size=tr("local_install_size_cuda" if nvidia else "local_install_size"),
              environment=environment_description())
    return message_dialog.ask(parent, tr("local_install_title"), text, tr("install_button"),
                              tr("dialog_cancel_button"), message_dialog.UPDATE)


class InstallDialog(QDialog):
    """Runs ``pip install`` for the given requirements; ``exec()`` returns Accepted after success."""

    def __init__(self, parent: Optional[QWidget], translator: TranslationManager, requirements: List[str]) -> None:
        """Start pip at once; the dialog shows its output until it ends."""
        super().__init__(parent)
        tr = translator.tr
        self._tr = tr
        self.setWindowTitle(tr("local_install_title"))
        self.resize(720, 420)
        layout = QVBoxLayout(self)
        self._label = QLabel(tr("local_install_running"), self)
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
        self._cancel = QPushButton(tr("local_install_cancel"), self)
        self._buttons.addButton(self._cancel, QDialogButtonBox.ButtonRole.RejectRole)
        self._cancel.clicked.connect(self._on_cancel)
        layout.addWidget(self._buttons)
        self.succeeded = False

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read_output)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_error)
        program, arguments = console_python(), pip_command(requirements)
        logging.info("local_engine_install start: %s %s", program, " ".join(arguments))
        self._output.appendPlainText(f"$ {program} {' '.join(arguments)}\n")
        self._process.start(program, arguments)

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
        logging.info("local_engine_install finished: exit code %s", code)
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        self._label.setText(self._tr("local_install_done" if self.succeeded else "local_install_failed", code=code))
        self._cancel.setText(self._tr("local_install_close"))
        self._cancel.clicked.disconnect()
        self._cancel.clicked.connect(self.accept if self.succeeded else self.reject)

    def _on_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            logging.error("local_engine_install could not start pip: %s", self._process.errorString())
            self._output.appendPlainText(self._process.errorString())
            self._on_finished(-1, QProcess.ExitStatus.CrashExit)

    def _on_cancel(self) -> None:
        if self._process.state() != QProcess.ProcessState.NotRunning:
            logging.info("local_engine_install cancelled by the user")
            self._process.kill()
            self._process.waitForFinished(3000)
        self.reject()

    def keyPressEvent(self, event: object) -> None:  # noqa: N802 - Qt API
        """Escape cancels like the button (stopping pip first)."""
        from PyQt6.QtGui import QKeyEvent

        if isinstance(event, QKeyEvent) and event.key() == Qt.Key.Key_Escape:
            self._on_cancel()
            return
        super().keyPressEvent(event)  # type: ignore[arg-type]


def install(parent: QWidget, translator: TranslationManager, requirements: List[str]) -> bool:
    """Confirm, then install ``requirements``; True when they are installed afterwards."""
    if not requirements:
        return True
    if not confirm_install(parent, translator, requirements):
        logging.info("local_engine_install declined by the user: %s", requirements)
        return False
    dialog = InstallDialog(parent, translator, requirements)
    dialog.exec()
    return dialog.succeeded
