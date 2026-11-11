"""Start with Windows: the Run key value (faked by conftest's ``run_key``) and the tray switch."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, Mock

import pytest

from app.platform import autostart

COMMAND = [r"C:\Program Files\Python\pythonw.exe", r"C:\Apps\WhisperTyper\run.py"]
COMMAND_LINE = r'"C:\Program Files\Python\pythonw.exe" C:\Apps\WhisperTyper\run.py'


def test_enable_registers_the_quoted_command_and_disable_removes_it(run_key):
    assert not autostart.is_enabled()
    autostart.enable(COMMAND)
    assert run_key == {"WhisperTyper": COMMAND_LINE}
    assert autostart.is_enabled()
    autostart.disable()
    assert run_key == {}
    autostart.disable()  # already off: nothing to remove


def test_refresh_follows_a_moved_installation_but_never_enables(run_key):
    assert autostart.refresh(COMMAND) is False
    assert run_key == {}
    run_key["WhisperTyper"] = r"C:\old\pythonw.exe C:\old\run.py"
    assert autostart.refresh(COMMAND) is True
    assert run_key == {"WhisperTyper": COMMAND_LINE}
    assert autostart.refresh(COMMAND) is False  # already current


def test_a_run_with_a_throwaway_home_folder_leaves_the_start_at_login_alone(monkeypatch, run_key):
    system_integration = pytest.importorskip("app.controllers.system_integration")  # needs Qt
    monkeypatch.setenv(system_integration.SKIP_ENV, "1")
    run_key["WhisperTyper"] = "elsewhere.exe"
    system_integration.SystemIntegration(Mock(config={})).apply()
    assert run_key == {"WhisperTyper": "elsewhere.exe"}


@pytest.fixture(scope="module")
def qapp():
    QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def tray(qapp, tmp_path, monkeypatch):
    """A built tray menu that offers the switch on every platform."""
    from PyQt6 import QtWidgets

    from app.audio.store import RecordingStore
    from app.context import AppContext
    from app.controllers import system_integration
    from app.controllers import tray as tray_module
    from app.core.config_store import ConfigStore
    from app.core.hotkeys import normalize_hotkey_string
    from app.ui import theme

    monkeypatch.setattr(autostart, "supported", lambda: True)
    monkeypatch.setattr(system_integration, "tray_command", lambda: COMMAND)
    monkeypatch.setattr(tray_module, "GitUpdater", lambda _parent: MagicMock(root=None))
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"ui_language": "en"}), encoding="utf-8")
    ctx = AppContext(ConfigStore(str(path), normalize_hotkey_string), RecordingStore(str(tmp_path)))
    ctx.notifier = Mock()
    settings = Mock(theme_palette=lambda: theme.palette(False))
    controller = tray_module.TrayController(
        ctx, QtWidgets.QWidget(), settings, recording=Mock(is_recording=False), pipeline=Mock(),
        files=Mock(log_file_exists=Mock(return_value=True)), restart=Mock(), quit_app=Mock())
    controller.build()
    yield controller
    controller.hide()


def _press_w(controller):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    QTest.keyClick(controller.tray_menu, Qt.Key.Key_W)


@pytest.mark.usefixtures("delete_windows_after_test")
def test_w_in_the_tray_menu_switches_the_start_at_login(tray, run_key):
    assert tray.autostart_action.text() == "Start with Windows"
    assert tray.autostart_action.icon().isNull()
    _press_w(tray)
    assert run_key == {"WhisperTyper": COMMAND_LINE}
    assert not tray.autostart_action.icon().isNull()  # the check mark
    _press_w(tray)
    assert run_key == {}
    assert tray.autostart_action.icon().isNull()


@pytest.mark.usefixtures("delete_windows_after_test")
def test_the_switch_shows_the_registry_state_when_the_menu_opens(tray, run_key):
    run_key["WhisperTyper"] = "added elsewhere"
    tray.tray_menu.aboutToShow.emit()
    assert not tray.autostart_action.icon().isNull()


@pytest.mark.usefixtures("delete_windows_after_test")
def test_a_failed_switch_is_reported(tray, monkeypatch):
    def refuse(_command_line):
        raise OSError("access denied")

    monkeypatch.setattr(autostart, "_write_run_value", refuse)
    _press_w(tray)
    message = tray._ctx.notifier.show.call_args.args[0]
    assert "access denied" in message


@pytest.fixture
def integration(monkeypatch):
    """apply_now with only the start at login left to do."""
    system_integration = pytest.importorskip("app.controllers.system_integration")  # needs Qt
    monkeypatch.setattr(autostart, "supported", lambda: True)
    monkeypatch.setattr(system_integration, "tray_command", lambda: COMMAND)
    monkeypatch.setattr(system_integration.command_shim, "supported", lambda: False)
    monkeypatch.setattr(system_integration.start_menu, "supported", lambda: False)
    return system_integration


def test_a_new_installation_starts_at_login(integration, run_key):
    assert integration.apply_now({"onboarding_done": False}) == []
    assert run_key == {"WhisperTyper": COMMAND_LINE}


def test_an_existing_installation_keeps_its_choice(integration, run_key):
    assert integration.apply_now({"onboarding_done": True}) == []
    assert run_key == {}


@pytest.mark.usefixtures("delete_windows_after_test")
def test_the_settings_switch_shares_the_registry_with_the_tray(qapp, tmp_path, monkeypatch, run_key):
    from app.audio.store import RecordingStore
    from app.context import AppContext
    from app.controllers import system_integration
    from app.core.config_store import ConfigStore
    from app.core.hotkeys import normalize_hotkey_string
    from app.ui.settings.window import SettingsWindow

    monkeypatch.setattr(autostart, "supported", lambda: True)
    monkeypatch.setattr(system_integration, "tray_command", lambda: COMMAND)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"ui_language": "en"}), encoding="utf-8")
    ctx = AppContext(ConfigStore(str(path), normalize_hotkey_string), RecordingStore(str(tmp_path)))
    window = SettingsWindow(ctx, recording=Mock(selectable_input_devices=Mock(return_value=[])), hotkeys=Mock(),
                            files=Mock(log_file_exists=Mock(return_value=False)), quit_app=Mock())
    switch = window.start_at_login_checkbox
    assert not switch.isChecked()
    switch.setChecked(True)
    assert run_key == {"WhisperTyper": COMMAND_LINE}
    run_key.clear()  # switched off in the tray menu meanwhile
    window.refresh_start_at_login_switch()
    assert not switch.isChecked()
    assert "start_at_login" not in json.loads(path.read_text(encoding="utf-8"))
