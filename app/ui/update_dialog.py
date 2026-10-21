"""Result dialog of the self-update: a message above git's output, which scrolls instead of growing.

A failed pull can list hundreds of files; in a message box they would push the buttons below
the screen edge, so the dialog never gets taller than the screen allows.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

from PyQt6.QtGui import QFontDatabase, QGuiApplication
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QPlainTextEdit, QVBoxLayout, QWidget

#: Share of the available screen height the dialog may use at most.
_MAX_SCREEN_SHARE = 0.85
#: Lines of output shown before the output box scrolls.
_PREFERRED_OUTPUT_LINES = 14

#: (caption, role) of a dialog button; ``run_update_dialog`` returns the caption that was clicked.
Button = Tuple[str, QDialogButtonBox.ButtonRole]


def run_update_dialog(parent: Optional[QWidget], title: str, message: str, output: str,
                      buttons: Sequence[Button], default: int = 0) -> Optional[str]:
    """Show ``message`` and the scrollable ``output``; the clicked button's caption, or None when closed."""
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    layout = QVBoxLayout(dialog)
    label = QLabel(message, dialog)
    label.setWordWrap(True)
    layout.addWidget(label)
    if output:
        text = QPlainTextEdit(output, dialog)
        text.setReadOnly(True)
        text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        text.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        lines = min(_PREFERRED_OUTPUT_LINES, output.count("\n") + 1)
        text.setMinimumHeight(text.fontMetrics().lineSpacing() * (lines + 1) + 12)
        layout.addWidget(text, 1)
    box = QDialogButtonBox(dialog)
    clicked: list[str] = []
    for index, (caption, role) in enumerate(buttons):
        button = box.addButton(caption, role)
        button.clicked.connect(lambda _checked=False, caption=caption: (clicked.append(caption), dialog.accept()))
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
