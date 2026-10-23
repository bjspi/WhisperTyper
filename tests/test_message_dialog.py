"""Message dialogs: the clicked caption, the default button, folded details and the parent's theme."""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PyQt6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication, QWidget  # noqa: E402

from app.ui import message_dialog, theme  # noqa: E402
from app.ui.message_dialog import DANGER, PRIMARY, SECONDARY, MessageDialog  # noqa: E402

QAPP = QApplication.instance() or QApplication([])


class _DarkWindow(QWidget):
    def theme_palette(self):
        return theme.palette(True, True)


def test_details_toggle_names_what_a_click_does():
    dialog = MessageDialog(None, "Title", "Text", [("Close", SECONDARY)], details="diff",
                           details_captions=("Show", "Hide"))
    dialog.show()
    assert dialog.details is not None and dialog.details_toggle is not None
    seen = [(dialog.details_toggle.text(), dialog.details.isVisible())]
    for _ in range(2):
        dialog.details_toggle.click()
        seen.append((dialog.details_toggle.text(), dialog.details.isVisible()))
    assert seen == [("Show", False), ("Hide", True), ("Show", False)]
    dialog.deleteLater()


def test_the_primary_button_is_the_default_and_details_without_toggle_show():
    dialog = MessageDialog(None, "Title", "Text", [("Cancel", SECONDARY), ("Delete", DANGER), ("Go", PRIMARY)],
                           details="output")
    defaults = [button.text() for button in dialog.buttons if button.isDefault()]
    assert defaults == ["Go"] and dialog.details_toggle is None
    assert not dialog.details.isHidden()  # type: ignore[union-attr]
    dialog.deleteLater()


def test_the_dialog_follows_the_parent_theme():
    parent = _DarkWindow()
    dialog = MessageDialog(parent, "Title", "Text", [("OK", PRIMARY)])
    assert theme.palette(True, True)["panel"] in dialog.styleSheet()
    dialog.deleteLater()
    parent.deleteLater()


def test_show_dialog_returns_the_clicked_caption_and_ask_a_bool():
    def click(caption):
        def run():
            dialog = next(w for w in QAPP.topLevelWidgets() if isinstance(w, MessageDialog) and w.isVisible())
            next(b for b in dialog.buttons if b.text() == caption).click()
        QTimer.singleShot(0, run)

    click("Retry")
    assert message_dialog.show_dialog(None, "T", "x", [("Close", SECONDARY), ("Retry", PRIMARY)]) == "Retry"
    click("Quit")
    assert message_dialog.ask(None, "Quit?", "", "Quit", "Cancel") is True
    click("Cancel")
    assert message_dialog.ask(None, "Quit?", "", "Quit", "Cancel") is False


def test_enter_cancels_a_destructive_question():
    dialog = MessageDialog(None, "Delete?", "", [("Cancel", SECONDARY), ("Delete", DANGER)])
    assert [button.text() for button in dialog.buttons if button.isDefault()] == ["Cancel"]
    dialog.deleteLater()
