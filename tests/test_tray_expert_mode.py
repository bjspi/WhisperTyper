"""Tray menu and expert mode: the expert tools leave the menu while it is off."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, Mock

import pytest

QtWidgets = pytest.importorskip("PyQt6.QtWidgets")

from app.audio.store import RecordingStore  # noqa: E402
from app.context import AppContext  # noqa: E402
from app.controllers import tray as tray_module  # noqa: E402
from app.core.config_store import ConfigStore  # noqa: E402
from app.core.expert_mode import expert_mode_shown  # noqa: E402
from app.core.hotkeys import normalize_hotkey_string  # noqa: E402
from app.ui import theme  # noqa: E402

pytestmark = pytest.mark.usefixtures("delete_windows_after_test")


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _make_tray(tmp_path, monkeypatch, *, expert_mode):
    """A built tray for the given expert mode; the updater never asks GitHub."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"ui_language": "en", "expert_mode": expert_mode}), encoding="utf-8")
    ctx = AppContext(ConfigStore(str(path), normalize_hotkey_string), RecordingStore(str(tmp_path)))
    monkeypatch.setattr(tray_module, "GitUpdater", lambda _parent: MagicMock(root=None))
    settings = Mock(theme_palette=lambda: theme.palette(False))
    controller = tray_module.TrayController(
        ctx, QtWidgets.QWidget(), settings, recording=Mock(is_recording=False), pipeline=Mock(),
        files=Mock(log_file_exists=Mock(return_value=True)), restart=Mock(), quit_app=Mock())
    controller.build()
    return controller


def _expert_tools(controller):
    return (controller.open_config_action, controller.open_log_action, controller.github_action,
            controller.mark_bad_action)


def test_expert_mode_off_hides_and_disables_the_expert_tools(qapp, tmp_path, monkeypatch):
    controller = _make_tray(tmp_path, monkeypatch, expert_mode=False)
    for action in _expert_tools(controller):
        assert not action.isVisible() and not action.isEnabled()  # a hidden badge key must not trigger it
    assert controller.play_action.isVisible() and controller.retranscribe_action.isVisible()


def test_the_menu_follows_expert_mode_when_it_opens(qapp, tmp_path, monkeypatch):
    controller = _make_tray(tmp_path, monkeypatch, expert_mode=False)
    controller._ctx.config["expert_mode"] = True
    controller.tray_menu.aboutToShow.emit()
    assert all(action.isVisible() for action in _expert_tools(controller))
    assert controller.open_log_action.isEnabled() and controller.github_action.isEnabled()


def test_expert_mode_defaults_to_shown():
    assert expert_mode_shown({})
    assert not expert_mode_shown({"expert_mode": False})
