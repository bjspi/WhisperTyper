"""Tray menu: a click right after the menu opens does not trigger the entry under the pointer."""
from __future__ import annotations

import pytest

QtWidgets = pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtCore import QPoint, Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402

from app.ui import theme  # noqa: E402
from app.ui.tray_menu import CLICK_GUARD_S, BadgeTrayMenu  # noqa: E402

pytestmark = pytest.mark.usefixtures("delete_windows_after_test")


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_a_stray_click_right_after_opening_does_not_quit(qapp):
    menu = BadgeTrayMenu(None, lambda: theme.palette(False))
    quits = []
    quit_action = menu.addAction("Quit")
    quit_action.triggered.connect(lambda: quits.append(True))
    menu.popup(QPoint(200, 200))
    QTest.qWaitForWindowExposed(menu)
    center = menu.actionGeometry(quit_action).center()

    QTest.mouseClick(menu, Qt.MouseButton.LeftButton, pos=center)
    assert not quits

    QTest.qWait(int(CLICK_GUARD_S * 1000) + 100)
    QTest.mouseClick(menu, Qt.MouseButton.LeftButton, pos=center)
    assert quits
