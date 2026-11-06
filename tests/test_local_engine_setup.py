"""Local transcription chosen in the guide: everything installs in the background, without a window."""
from __future__ import annotations

import os
from unittest.mock import Mock

import pytest

pytest.importorskip("PyQt6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QCoreApplication, QEventLoop, QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from app.controllers import local_engine as local_engine_module  # noqa: E402
from app.controllers.local_engine import LocalEngineController  # noqa: E402
from app.local_engine.supervisor import READY, EngineStatus  # noqa: E402

QAPP = QApplication.instance() or QApplication([])  # must stay referenced for the whole run


@pytest.fixture
def engine(monkeypatch):
    """A controller with a pending setup; nothing real is installed or started."""
    ctx = Mock(config={"local_engine_setup_pending": True, "local_engine_enabled": True}, tr=lambda key, **_kw: key)
    controller = LocalEngineController(ctx)
    controller.detect_devices = Mock()
    monkeypatch.setattr(local_engine_module.availability, "display_adapters", lambda: [])
    monkeypatch.setattr(local_engine_module.availability, "install_packages", lambda _adapters: ["faster-whisper"])
    monkeypatch.setattr(local_engine_module.availability, "engine_installed", lambda: True)
    yield controller
    controller.shutdown()


def wait_for_setup(controller: LocalEngineController) -> None:
    """Run the event loop until the installer ended."""
    loop = QEventLoop()
    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: loop.quit() if controller._setup_process is None else None)
    timer.start()
    QTimer.singleShot(20000, loop.quit)
    loop.exec()
    QCoreApplication.processEvents()


def installer(monkeypatch, exit_code: int) -> None:
    """Replace pip by a small process that ends with ``exit_code``."""
    import sys
    command = [sys.executable, "-c", f"print('fake pip'); raise SystemExit({exit_code})"]
    monkeypatch.setattr(local_engine_module.installation, "install_command", lambda _requirements: command)


def test_a_pending_setup_blocks_transcription_with_a_clear_reason_and_keeps_the_engine_down(engine):
    assert engine.problem() == "local_engine_setting_up"
    engine._start = Mock()
    engine.apply_config()
    engine._start.assert_not_called()


def test_a_successful_setup_loads_the_engine_and_announces_it_once(engine, monkeypatch):
    installer(monkeypatch, 0)
    engine.setup_in_background()
    assert engine._setup_process is not None
    wait_for_setup(engine)
    assert engine._ctx.config["local_engine_setup_pending"] is False
    engine._ctx.save_config.assert_called()
    assert engine._start_after_detection
    engine.detect_devices.assert_called_once()
    shown = [call.args[0] for call in engine._ctx.notifier.show.call_args_list]
    assert shown == ["local_engine_setup_model"]

    for _status in range(2):
        engine._on_status(EngineStatus(state=READY))
    assert [call.args[0] for call in engine._ctx.notifier.show.call_args_list].count("local_engine_setup_ready") == 1


def test_a_failed_setup_says_so_and_leaves_the_engine_down(engine, monkeypatch):
    installer(monkeypatch, 3)
    engine.setup_in_background()
    wait_for_setup(engine)
    assert engine._ctx.config["local_engine_setup_pending"] is False
    engine.detect_devices.assert_not_called()
    engine._ctx.notifier.show.assert_called_once()
    assert engine._ctx.notifier.show.call_args.args[0] == "local_engine_setup_failed"


def test_nothing_to_install_goes_straight_to_the_engine(engine, monkeypatch):
    monkeypatch.setattr(local_engine_module.availability, "install_packages", lambda _adapters: [])
    engine.setup_in_background()
    assert engine._setup_process is None and engine._ctx.config["local_engine_setup_pending"] is False
    engine.detect_devices.assert_called_once()


def test_quitting_during_the_setup_keeps_it_pending_for_the_next_start(engine, monkeypatch):
    import sys
    command = [sys.executable, "-c", "import time; time.sleep(30)"]
    monkeypatch.setattr(local_engine_module.installation, "install_command", lambda _requirements: command)
    engine.setup_in_background()
    engine.shutdown()
    assert engine._ctx.config["local_engine_setup_pending"] is True
