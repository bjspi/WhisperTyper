"""Shared test hygiene: windows a test module leaves open are deleted when the module ends.

Without this, every settings window, balloon and palette a module creates stays alive for the
rest of the run (hundreds of windows, ~14,000 widgets), and each later stylesheet change has to
repolish all of them, which made the Qt tests slower and slower. Module scope keeps the windows
of module-scoped fixtures alive while their tests run.
"""
from __future__ import annotations

from typing import Iterator

import pytest

try:
    from PyQt6.QtCore import QCoreApplication, QEvent
    from PyQt6.QtWidgets import QApplication
except ImportError:  # The lightweight CI matrix runs without Qt.
    QApplication = None  # type: ignore[assignment,misc]


def _reset_singletons() -> None:
    """Forget the shared overlay/balloon instances, whose windows were just deleted."""
    from app.ui.floating_buttons import FloatingButtonWindow, RecordingPromptOverlay
    from app.ui.tooltip import MouseFollowerTooltip

    for cls in (FloatingButtonWindow, RecordingPromptOverlay, MouseFollowerTooltip):
        cls._instance = None


@pytest.fixture(autouse=True, scope="module")
def _delete_leftover_windows() -> Iterator[None]:
    app = QApplication.instance() if QApplication is not None else None
    before = set(map(id, app.topLevelWidgets())) if app is not None else set()
    yield
    app = QApplication.instance() if QApplication is not None else None
    if app is None:
        return
    leftovers = [widget for widget in app.topLevelWidgets() if id(widget) not in before]
    for widget in leftovers:
        widget.close()
        widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    app.processEvents()
    _reset_singletons()
