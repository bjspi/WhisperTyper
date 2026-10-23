"""Dialogs of the self-update: a message above optional details (changes, git's output) that scroll.

The details stay folded behind a toggle button, as most users only care about the version. A
release diff or a failed update can run to hundreds of lines; in a message box they would push the
buttons below the screen edge, so the dialog never gets taller than the screen allows.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

from PyQt6.QtGui import QFontDatabase, QGuiApplication
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

#: Share of the available screen height the dialog may use at most.
_MAX_SCREEN_SHARE = 0.85
#: Lines of output shown before the output box scrolls.
_PREFERRED_OUTPUT_LINES = 14

#: (caption, role) of a dialog button; ``run_update_dialog`` returns the caption that was clicked.
Button = Tuple[str, QDialogButtonBox.ButtonRole]


def run_update_dialog(parent: Optional[QWidget], title: str, message: str, output: str,
                      buttons: Sequence[Button], default: int = 0,
                      details_caption: Optional[str] = None) -> Optional[str]:
    """Show ``message`` and the scrollable ``output``; the clicked button's caption, or None when closed.

    With ``details_caption`` the output starts hidden behind a toggle button of that caption.
    """
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    layout = QVBoxLayout(dialog)
    label = QLabel(message, dialog)
    label.setWordWrap(True)
    layout.addWidget(label)
    text: Optional[QPlainTextEdit] = None
    if output:
        text = QPlainTextEdit(output, dialog)
        text.setReadOnly(True)
        text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        text.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        lines = min(_PREFERRED_OUTPUT_LINES, output.count("\n") + 1)
        text.setMinimumHeight(text.fontMetrics().lineSpacing() * (lines + 1) + 12)
        layout.addWidget(text, 1)
    box = QDialogButtonBox(dialog)
    if text is not None and details_caption:
        details = text
        details.setVisible(False)
        toggle = QPushButton(details_caption, dialog)
        toggle.setCheckable(True)
        toggle.setAutoDefault(False)

        def show_details(shown: bool) -> None:
            details.setVisible(shown)
            dialog.adjustSize()

        toggle.toggled.connect(show_details)
        box.addButton(toggle, QDialogButtonBox.ButtonRole.HelpRole)
    clicked: list[str] = []

    def choose(caption: str) -> None:
        clicked.append(caption)
        dialog.accept()

    for index, (caption, role) in enumerate(buttons):
        button = box.addButton(caption, role)
        if button is None:
            continue
        button.clicked.connect(lambda _checked=False, caption=caption: choose(caption))
        if index == default:
            button.setDefault(True)
    layout.addWidget(box)

    dialog.setMinimumWidth(520)
    screen = (parent.screen() if parent is not None else None) or QGuiApplication.primaryScreen()
    if screen is not None:
        available = screen.availableGeometry()
        dialog.setMaximumHeight(int(available.height() * _MAX_SCREEN_SHARE))
        dialog.setMaximumWidth(int(available.width() * _MAX_SCREEN_SHARE))
    dialog.exec()
    return clicked[0] if clicked else None
