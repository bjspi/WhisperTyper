"""Local transcription engine on the GUI side: when it runs, with which settings, and its status.

The engine runs only while it is enabled and needed: the "Local" provider is selected for
transcription, or the server for other programs is on. Then it is loaded at startup ("preload")
or for the first recording ("on demand"), and unloaded after the idle time (never while the
server is on). The packages are optional: enabling the engine offers to install them.
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import QObject, QProcess, pyqtSignal

from app.context import AppContext
from app.core.api_keys import LOCAL_PROVIDER, provider_for_url
from app.core.constants import APP_DATA_DIR
from app.core.env import is_MACOS
from app.local_engine import availability
from app.local_engine.catalog import (
    AUTO_DEVICE,
    DEFAULT_SERVER_PORT,
    EngineSettings,
    backend_of,
    resolve_device,
    settings_from_config,
)
from app.local_engine.supervisor import OFF, PROJECT_ROOT, EngineStatus, Supervisor, free_port

#: The engine cannot run: packages missing, no usable device, or a frozen (PyInstaller) build.
UNAVAILABLE = "unavailable"
#: The local engine is offered on Windows and Linux. On macOS it is switched off entirely (no page,
#: no "Local" provider, no tray menu): its GPU backends (CUDA, Intel OpenVINO) do not exist there,
#: the app bundle cannot start ``python -m`` child processes, and it is untested on Macs. An Apple
#: GPU backend (e.g. mlx-whisper) would be the way to offer it there.
LOCAL_ENGINE_SUPPORTED = not is_MACOS
#: Seconds a recording waits for the engine to finish loading before it reports an error.
READY_WAIT_S = 120.0


def local_selected(config: Dict[str, Any]) -> bool:
    """The transcription provider is the local engine."""
    return provider_for_url(str(config.get("api_endpoint", ""))) == LOCAL_PROVIDER


class LocalEngineController(QObject):
    """Decide when the engine runs and expose its status to the settings page and the tray."""

    #: A new :class:`EngineStatus` (GUI thread).
    status_changed = pyqtSignal(object)
    #: The device list was (re)detected.
    devices_changed = pyqtSignal()
    _status_from_thread = pyqtSignal(object)

    def __init__(self, ctx: AppContext) -> None:
        """Nothing runs until ``apply_config`` or a recording needs the engine."""
        super().__init__()
        self._ctx = ctx
        self._config = ctx.config
        self._supervisor = Supervisor(on_change=self._status_from_thread.emit)
        self._status_from_thread.connect(self._on_status)
        self._status = EngineStatus()
        self._internal_port: Optional[int] = None
        #: Detected devices ``[{"id": ..., "name": ...}]``; None until the first detection finished.
        self.devices: Optional[List[Dict[str, str]]] = None
        self.device_errors: Dict[str, str] = {}
        self._detector: Optional[QProcess] = None
        #: The device detection is running.
        self.detecting = False
        #: Start the engine once the device detection is done ("auto" needs its result).
        self._start_after_detection = False

    # --- State ----------------------------------------------------------------------------
    @property
    def status(self) -> EngineStatus:
        """The latest engine status."""
        return self._status

    def enabled(self) -> bool:
        """Local transcription is switched on in the saved settings (never on macOS)."""
        return LOCAL_ENGINE_SUPPORTED and bool(self._config.get("local_engine_enabled", False))

    def server_enabled(self) -> bool:
        """The OpenAI-compatible endpoint for other programs is on."""
        return self.enabled() and bool(self._config.get("local_server_enabled", False))

    def needed(self) -> bool:
        """The engine has a job: local transcription is selected, or the server is on."""
        return self.enabled() and (local_selected(self._config) or self.server_enabled())

    @staticmethod
    def supported() -> bool:
        """Not on macOS, and not in a frozen build (the engine runs as ``python -m``)."""
        return LOCAL_ENGINE_SUPPORTED and not getattr(sys, "frozen", False)

    def device(self) -> Optional[str]:
        """The concrete device for the configured choice (None while "auto" waits for the detection)."""
        choice = str(self._config.get("local_device") or AUTO_DEVICE)
        if choice != AUTO_DEVICE:
            return resolve_device(choice, [])
        if self.devices is None:
            return None
        return resolve_device(choice, [device["id"] for device in self.devices])

    def settings(self) -> Optional[EngineSettings]:
        """What the engine runs with now, or None without a usable device."""
        device = self.device()
        if device is None:
            return None
        return settings_from_config(self._config, device, APP_DATA_DIR, self._port())

    def endpoint(self) -> str:
        """Transcription URL of the running (or starting) engine.

        Known before the device is (the address does not depend on it), so a recording that ends
        while the device detection still runs already knows where to send.
        """
        settings = self._supervisor.settings or self.settings()
        if settings is None:
            settings = settings_from_config(self._config, "cpu", APP_DATA_DIR, self._port())
        return settings.endpoint

    def _port(self) -> int:
        """The configured server port, or a free loopback port kept for this session."""
        if self.server_enabled():
            return int(self._config.get("local_server_port") or DEFAULT_SERVER_PORT)
        if self._internal_port is None:
            self._internal_port = free_port()
        return self._internal_port

    def problem(self) -> str:
        """Translation key of why a local transcription cannot run now, or ""."""
        return "local_engine_disabled" if not self.enabled() else self.unavailable_reason()

    def unavailable_reason(self) -> str:
        """Translation key of why the engine cannot run, or "" when it can."""
        if not self.supported():
            return "local_engine_unsupported"
        if not availability.engine_installed():
            return "local_engine_packages_missing"
        device = self.device()
        if device is None:
            return "" if self.devices is None else "local_engine_no_device"
        if not availability.backend_ready(backend_of(device)):
            return "local_engine_packages_missing"
        return ""

    # --- Lifecycle ------------------------------------------------------------------------
    def apply_config(self) -> None:
        """Start, restart or stop the engine to match the saved settings."""
        logging.info("local_engine_apply enabled=%s local_selected=%s server=%s strategy=%s idle_min=%s device=%s",
                     self.enabled(), local_selected(self._config), self.server_enabled(),
                     self._config.get("local_load_strategy"), self._config.get("local_idle_unload_minutes"),
                     self._config.get("local_device"))
        if not self.needed():
            self._start_after_detection = False
            if self._supervisor.wanted or self._status.state != OFF:
                self._supervisor.set_target(None)
            return
        server = self.server_enabled()
        minutes = 0 if server else max(0, int(self._config.get("local_idle_unload_minutes", 0) or 0))
        self._supervisor.idle_unload_s = minutes * 60.0
        preload = server or self._config.get("local_load_strategy", "preload") == "preload"
        if preload or self._supervisor.wanted:
            self._start()

    def ensure_running(self) -> None:
        """A local transcription is about to run (recording started): load the engine if it is not."""
        if not (self.enabled() and local_selected(self._config)):
            return
        self._supervisor.touch()
        if not self._supervisor.wanted:
            logging.info("local_engine_load reason=recording")
            self._start()

    def touch(self) -> None:
        """Restart the idle timer (a local transcription just ran)."""
        self._supervisor.touch()

    def _start(self) -> None:
        if self.device() is None and self.devices is None:
            self._start_after_detection = True
            self.detect_devices()
            return
        reason = self.unavailable_reason()
        settings = self.settings() if not reason else None
        if settings is None:
            logging.warning("local_engine_unavailable reason=%s devices=%s errors=%s", reason or "no device",
                            self.devices, self.device_errors)
            self._supervisor.set_target(None)
            self._on_status(EngineStatus(state=UNAVAILABLE, detail=reason or "local_engine_no_device"))
            return
        logging.info("local_engine_target %s", settings.to_json())
        self._supervisor.start_thread()
        self._supervisor.set_target(settings)

    def stop(self) -> None:
        """Unload now (tray / settings); the next local recording loads it again."""
        logging.info("local_engine_unload reason=user")
        self._supervisor.set_target(None, reason="stopped")

    def restart(self) -> None:
        """Restart a running engine, or load a stopped one."""
        if self._supervisor.wanted:
            self._supervisor.restart()
        else:
            self._start()

    def self_test(self) -> None:
        """Run the deep check (a tiny inference) now."""
        logging.info("local_engine_self_test requested by the user")
        self._supervisor.request_deep_check()

    def shutdown(self) -> None:
        """Stop the engine process (app quit)."""
        detector, self._detector = self._detector, None
        if detector is not None:
            # A QProcess destroyed while its child still runs aborts the app at exit.
            detector.kill()
            detector.waitForFinished(3000)
        self._supervisor.shutdown()

    def _on_status(self, status: EngineStatus) -> None:
        self._status = status
        self.status_changed.emit(status)

    # --- Devices --------------------------------------------------------------------------
    def detect_devices(self) -> None:
        """Ask a helper process which devices the installed backends see (OpenVINO and CUDA stay out of the app)."""
        if self._detector is not None:
            return
        if not self.supported() or not availability.engine_installed():
            self.devices, self.device_errors = [], {}
            self.devices_changed.emit()
            self._detection_finished()
            return
        process = QProcess(self)
        process.setProgram(availability.console_python())
        process.setArguments(["-m", "app.local_engine.devices"])
        process.setWorkingDirectory(PROJECT_ROOT)
        process.finished.connect(lambda _code, _status: self._on_detected(process))
        process.errorOccurred.connect(lambda _error: self._on_detected(process))
        self._detector = process
        self.detecting = True
        logging.info("local_engine_devices detecting")
        process.start()
        self.devices_changed.emit()  # pages show "detecting…"

    def _on_detected(self, process: QProcess) -> None:
        if self._detector is not process:
            return
        self._detector = None
        self.detecting = False
        output = bytes(process.readAllStandardOutput().data()).decode("utf-8", "replace").strip()
        try:
            result = json.loads(output.splitlines()[-1]) if output else {}
        except (ValueError, IndexError):
            result = {}
        self.devices = [device for device in result.get("devices", []) if isinstance(device, dict)]
        self.device_errors = dict(result.get("errors", {}))
        if not output:
            self.device_errors["detection"] = bytes(process.readAllStandardError().data()).decode(
                "utf-8", "replace").strip()[-500:]
        logging.info("local_engine_devices %s errors=%s", [d.get("id") for d in self.devices], self.device_errors)
        process.deleteLater()
        self.devices_changed.emit()
        self._detection_finished()

    def _detection_finished(self) -> None:
        """Start the engine that waited for the device list."""
        if self._start_after_detection:
            self._start_after_detection = False
            if self.needed() or self._supervisor.wanted:
                self._start()
