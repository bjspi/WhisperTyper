"""Restarting the app: a new instance that cannot launch must leave this one running."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("PyQt6.QtWidgets")

from app import application  # noqa: E402
from app.application import WhisperTyperApp  # noqa: E402


def test_a_failed_launch_keeps_this_instance_running(monkeypatch):
    """Regression: PyQt6 returns ``(False, -1)``, a truthy tuple, so the failure quit the app."""
    monkeypatch.setattr(application.QProcess, "startDetached", lambda *_args: (False, -1))
    quit_app = Mock()
    monkeypatch.setattr(application.QApplication, "instance", lambda: Mock(quit=quit_app))
    app = SimpleNamespace(settings=Mock(isVisible=Mock(return_value=False)), hotkeys=Mock(), recording=Mock(),
                          local_engine=Mock(), ctx=Mock())
    WhisperTyperApp.restart_app(app)  # type: ignore[arg-type]
    quit_app.assert_not_called()
    app.local_engine.shutdown.assert_not_called()
    app.hotkeys.restart.assert_called_once()
    app.recording.apply_settings.assert_called_once()
