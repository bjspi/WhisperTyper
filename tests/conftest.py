"""Shared test hygiene: windows a test module leaves open are deleted when the module ends.

Without this, every settings window, balloon and palette a module creates stays alive for the
rest of the run (hundreds of windows, ~14,000 widgets), and each later stylesheet change has to
repolish all of them, which made the Qt tests slower and slower. Module scope keeps the windows
of module-scoped fixtures alive while their tests run.
"""
from __future__ import annotations

import importlib.util
from typing import Iterator, Set

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


@pytest.fixture(autouse=True)
def _keep_real_microphones_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A test take must never open this computer's input devices (tests run the scout directly)."""
    if importlib.util.find_spec("pyaudio") is None:  # The lightweight CI matrix runs without PyAudio.
        return
    from app.audio.mic_scout import MicrophoneScout

    monkeypatch.setattr(MicrophoneScout, "start", lambda _self: None)


def _windows_now() -> Set[int]:
    """IDs of the open top-level windows (none without a Qt application)."""
    app = QApplication.instance() if QApplication is not None else None
    return set(map(id, app.topLevelWidgets())) if app is not None else set()


def _delete_windows_opened_since(before: Set[int]) -> None:
    """Close and really delete every top-level window that is not in ``before``."""
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


@pytest.fixture(autouse=True, scope="module")
def _delete_leftover_windows() -> Iterator[None]:
    before = _windows_now()
    yield
    _delete_windows_opened_since(before)


@pytest.fixture
def delete_windows_after_test() -> Iterator[None]:
    """Opt-in for modules that build a settings window per test: each test's windows go right away."""
    before = _windows_now()
    yield
    _delete_windows_opened_since(before)
