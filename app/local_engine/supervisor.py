"""Supervisor of the engine process: starts and stops it, checks its health, restarts it when needed.

Runs its loop on its own thread (no Qt). Only that thread starts or stops processes; callers queue
requests and read :attr:`Supervisor.status`. Self-healing as in a long-running service:

* ``/health`` every few seconds; a crashed or unresponsive process is restarted with backoff.
* a tiny inference (``/health?deep=1``) every 15 minutes and right after standby, which catches a
  GPU that hangs or returns garbage after resume.
* an engine left over by a crashed app (orphan) is stopped; a foreign program on the port is a conflict.
* idle unload: after the configured time without use the process ends and frees the GPU memory.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.local_engine.catalog import SETTINGS_ENV, EngineSettings

log = logging.getLogger("whispertyper.engine")

OFF, STARTING, LOADING, READY, DOWN, CONFLICT = "off", "starting", "loading", "ready", "down", "conflict"
SERVER_MODULE = "app.local_engine.server"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Direct connection: a system proxy must never see (or break) localhost traffic.
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@dataclasses.dataclass(frozen=True)
class EngineStatus:
    """A snapshot for the UI. ``detail`` is a technical note (error text, conflicting program)."""

    state: str = OFF
    detail: str = ""
    #: Why the engine is off: "" (not needed), "idle" (unloaded after the idle time), "stopped" (by the user).
    reason: str = ""
    device: str = ""
    model: str = ""
    endpoint: str = ""
    #: Seconds of the last deep check (tiny inference), None before the first one.
    latency_s: Optional[float] = None
    downloaded_bytes: int = 0
    download_total_bytes: int = 0
    #: Health checked at (local time, HH:MM:SS).
    checked_at: str = ""
    #: Model downloading (only meaningful while ``state`` is LOADING).
    downloading: bool = False


def get_health(base_url: str, timeout: float, deep: bool = False) -> Dict[str, Any]:
    """The engine's /health answer (``deep`` runs a tiny inference)."""
    url = f"{base_url}/health" + ("?deep=1" if deep else "")
    with _opener.open(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def free_port() -> int:
    """A currently unused TCP port on the loopback interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def backoff_remaining(attempts: int, last_start: Optional[float], now: float, steps: Tuple[float, ...]) -> float:
    """Seconds until the next automatic restart may happen."""
    if not attempts or last_start is None:
        return 0.0
    delay = steps[min(attempts, len(steps) - 1)]
    return max(0.0, last_start + delay - now)


def server_command() -> List[str]:
    """The engine process command; pythonw avoids a console window."""
    exe = sys.executable
    pythonw = os.path.join(os.path.dirname(exe), "pythonw.exe")
    return [pythonw if sys.platform == "win32" and os.path.isfile(pythonw) else exe, "-m", SERVER_MODULE]


def is_engine_process(cmdline: List[str]) -> bool:
    """A process started as the engine (by this or an earlier, crashed app instance)."""
    args = [arg.lower() for arg in cmdline]
    return any(arg == "-m" and nxt == SERVER_MODULE for arg, nxt in zip(args, args[1:]))


def _child_environment(settings: EngineSettings) -> Dict[str, str]:
    env = dict(os.environ)
    env[SETTINGS_ENV] = settings.to_json()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [PROJECT_ROOT, env.get("PYTHONPATH", "")]))
    # The engine only talks to localhost; Hugging Face downloads still use the system proxy.
    env.setdefault("NO_PROXY", "127.0.0.1,localhost")
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _stop_tree(pid: int, timeout: float) -> None:
    """Terminate a process and its children, killing what does not end in time."""
    try:
        import psutil
    except ImportError:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return
    try:
        root = psutil.Process(pid)
        procs = [root] + root.children(recursive=True)
    except psutil.Error:
        return
    for proc in procs:
        try:
            proc.terminate()
        except psutil.Error:
            pass
    _gone, alive = psutil.wait_procs(procs, timeout=timeout)
    for proc in alive:
        try:
            proc.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(alive, timeout=timeout)


def _tree_pids(pid: int) -> set:
    try:
        import psutil

        root = psutil.Process(pid)
        return {pid} | {child.pid for child in root.children(recursive=True)}
    except Exception:  # noqa: BLE001 - psutil missing or the process is gone
        return {pid}


def _port_owner(port: int) -> Tuple[Optional[int], str, bool]:
    """(PID, name, is an engine process) of the program listening on ``port``, or (None, "", False)."""
    try:
        import psutil

        for conn in psutil.net_connections(kind="tcp"):
            if conn.status == psutil.CONN_LISTEN and conn.laddr and conn.laddr.port == port and conn.pid:
                proc = psutil.Process(conn.pid)
                return conn.pid, proc.name(), is_engine_process(proc.cmdline())
    except Exception as exc:  # noqa: BLE001 - psutil missing, access denied
        log.debug("Port owner lookup failed: %s", exc)
    return None, "", False


class Supervisor:
    """Owns the engine process and keeps it healthy; all process work happens on its thread."""

    check_interval_s = 10.0
    #: Faster checks while the engine starts, so the status follows the loading closely.
    starting_interval_s = 1.0
    deep_check_interval_s = 900.0
    http_timeout_s = 5.0
    deep_timeout_s = 60.0
    #: The first start downloads the model (0.5-3 GB).
    startup_timeout_s = 1800.0
    fail_threshold = 3
    restart_backoff_s: Tuple[float, ...] = (0, 10, 30, 60, 120, 300)
    stop_timeout_s = 10.0

    def __init__(self, on_change: Callable[[EngineStatus], None] = lambda status: None) -> None:
        """``on_change`` receives every new status, on the supervisor thread."""
        self.on_change = on_change
        self.status = EngineStatus()
        self.settings: Optional[EngineSettings] = None
        #: Seconds without use after which the process ends (0 = never).
        self.idle_unload_s = 0.0
        self._proc: Optional[subprocess.Popen] = None
        self._started_at: Optional[float] = None
        self._attempts = 0
        self._last_auto_start: Optional[float] = None
        self._fail_streak = 0
        #: Monotonic time of the last deep check; -inf runs the first one as soon as the engine is ready
        #: (``time.monotonic`` may be smaller than the interval shortly after boot).
        self._last_deep = float("-inf")
        self._deep_due = False
        self._last_used = time.monotonic()
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._lock = threading.Lock()
        self._target: Optional[EngineSettings] = None
        self._target_reason = ""
        self._target_changed = False
        self._restart_requested = False
        self._thread: Optional[threading.Thread] = None

    # --- Thread-safe API ------------------------------------------------------------------
    def start_thread(self) -> None:
        """Start the supervisor thread (once)."""
        if self._thread is None:
            self._thread = threading.Thread(target=self.run, name="LocalEngineSupervisor", daemon=True)
            self._thread.start()

    def set_target(self, settings: Optional[EngineSettings], reason: str = "") -> None:
        """Run the engine with ``settings`` (restarting it when they differ), or stop it (None)."""
        with self._lock:
            self._target, self._target_reason, self._target_changed = settings, reason, True
            if settings is not None:
                self._last_used = time.monotonic()
        self._wake.set()

    def restart(self) -> None:
        """Restart the engine process now."""
        with self._lock:
            self._restart_requested = True
        self._wake.set()

    def request_deep_check(self) -> None:
        """Run the deep check (a tiny inference) at the next tick."""
        self._deep_due = True
        self._wake.set()

    def touch(self) -> None:
        """The engine was just used (or is about to be): restart the idle timer."""
        self._last_used = time.monotonic()

    def shutdown(self, wait: bool = True) -> None:
        """End the loop and stop the engine process."""
        self._stopping.set()
        self._wake.set()
        if wait and self._thread is not None:
            self._thread.join(self.stop_timeout_s + 5)

    @property
    def wanted(self) -> bool:
        """Whether the engine should run."""
        return self._target is not None

    # --- Process (supervisor thread only) -------------------------------------------------
    def _alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _start(self, reason: str) -> None:
        assert self.settings is not None
        cmd = server_command()
        log.info("Starting the local engine (%s): %s on %s:%d", reason, self.settings.device, self.settings.host,
                 self.settings.port)
        self._started_at = time.monotonic()
        self._fail_streak = 0
        try:
            self._proc = subprocess.Popen(
                cmd, cwd=PROJECT_ROOT, env=_child_environment(self.settings), stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            log.info("Local engine started, PID %d", self._proc.pid)
        except OSError as exc:
            self._proc = None
            log.error("Starting the local engine failed: %s", exc)

    def _stop(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            log.info("Stopping the local engine (PID %d)", self._proc.pid)
            _stop_tree(self._proc.pid, self.stop_timeout_s)
        self._proc = None
        self._started_at = None

    def _restart(self, reason: str) -> bool:
        """Restart with backoff; False while the backoff is still running."""
        wait = backoff_remaining(self._attempts, self._last_auto_start, time.monotonic(), self.restart_backoff_s)
        if wait > 0:
            return False
        self._stop()
        self._start(f"{reason}, attempt {self._attempts + 1}")
        self._attempts += 1
        self._last_auto_start = time.monotonic()
        return True

    # --- Loop -----------------------------------------------------------------------------
    def run(self) -> None:
        """The supervisor loop; the engine process stops when it ends."""
        last_wall = time.time()
        try:
            while not self._stopping.is_set():
                self._wake.clear()
                gap = time.time() - last_wall
                if gap > self.check_interval_s + 30 and self._alive():
                    log.info("Clock jumped %.0f s (standby/hibernate?): deep check", gap)
                    self._deep_due = True
                last_wall = time.time()
                try:
                    self._handle_requests()
                    if not self._stopping.is_set():
                        self.tick()
                except Exception:  # noqa: BLE001 - the supervisor must never die
                    log.exception("Local engine supervisor tick failed")
                busy = self.status.state in (STARTING, LOADING)
                self._wake.wait(self.starting_interval_s if busy else self.check_interval_s)
        finally:
            self._stop()

    def _handle_requests(self) -> None:
        with self._lock:
            changed, target, reason = self._target_changed, self._target, self._target_reason
            restart, self._restart_requested, self._target_changed = self._restart_requested, False, False
        if changed and target is None:
            self._stop()
            self.settings = None
            self._set(OFF, reason=reason)
            return
        if changed and target != self.settings:
            if self._alive():
                log.info("Local engine settings changed: restarting")
            self.settings = target
            restart = True
        if restart and self.settings is not None:
            self._attempts, self._last_auto_start = 0, None
            self._stop()
            self._last_deep, self._deep_due = float("-inf"), False
            self._start("requested")
            self._set(STARTING)

    def _set(self, state: str, detail: str = "", reason: str = "", health: Optional[Dict[str, Any]] = None,
             latency: Optional[float] = None) -> None:
        settings = self.settings
        health = health or {}
        status = EngineStatus(
            state=state, detail=detail, reason=reason,
            device=settings.device if settings else "", model=settings.model if settings else "",
            endpoint=settings.endpoint if settings else "",
            latency_s=latency if latency is not None else (self.status.latency_s if state == READY else None),
            downloaded_bytes=int(health.get("downloaded_bytes", 0)),
            download_total_bytes=int(health.get("download_total_bytes", 0)),
            checked_at=time.strftime("%H:%M:%S"), downloading=health.get("status") == "downloading")
        if (status.state, status.detail) != (self.status.state, self.status.detail):
            log.info("Local engine: %s %s", state, detail)
        self.status = status
        self.on_change(status)

    def tick(self) -> None:
        """One health check: restart on crash, hang or model error; idle unload."""
        settings = self.settings
        if settings is None:
            return
        if self._proc is not None and not self._alive():
            log.warning("Local engine (PID %d) exited with code %s", self._proc.pid, self._proc.returncode)
            self._proc, self._started_at = None, None

        if not self._alive():
            owner, name, ours = _port_owner(settings.port)
            if owner is not None and ours:
                log.info("Stopping an orphaned local engine, PID %d on port %d", owner, settings.port)
                _stop_tree(owner, self.stop_timeout_s)
                owner = None
            if owner is not None:
                self._set(CONFLICT, f"{name} (PID {owner}), port {settings.port}")
                return
            if self._restart("not running"):
                self._set(STARTING)
            else:
                self._set(DOWN, self.status.detail or "not running")
            return

        since = time.monotonic() - (self._started_at or time.monotonic())
        try:
            health = get_health(settings.base_url, self.http_timeout_s)
        except Exception as exc:  # noqa: BLE001
            if since < 60:
                self._set(STARTING)
                return
            self._fail_streak += 1
            log.warning("Local engine health check failure #%d: %s", self._fail_streak, exc)
            if self._fail_streak >= self.fail_threshold:
                self._restart("not responding")
                self._set(DOWN, "not responding")
            return

        if self._proc is not None and health.get("pid") not in _tree_pids(self._proc.pid):
            self._set(CONFLICT, f"PID {health.get('pid')}, port {settings.port}")
            return
        self._fail_streak = 0
        status = health.get("status")
        if status in ("downloading", "loading"):
            if since > self.startup_timeout_s:
                self._restart("startup timeout")
                self._set(DOWN, f"{status} took longer than {self.startup_timeout_s:.0f} s")
            else:
                self._set(LOADING, health=health)
            return
        if status != "ready":
            error = str(health.get("error", status))[:300]
            self._restart("model error")
            self._set(DOWN, error)
            return

        if self.idle_unload_s > 0 and time.monotonic() - self._last_used > self.idle_unload_s:
            log.info("Local engine idle for %.0f min: unloading", self.idle_unload_s / 60)
            with self._lock:
                if not self._target_changed:
                    self._target = None
            self._stop()
            self.settings = None
            self._set(OFF, reason="idle")
            return

        latency = None
        if self._deep_due or time.monotonic() - self._last_deep > self.deep_check_interval_s:
            try:
                deep = get_health(settings.base_url, self.deep_timeout_s, deep=True)
                latency = deep.get("deep_s")
                self._last_deep, self._deep_due = time.monotonic(), False
                log.info("Local engine deep check OK: inference %.2f s", latency or 0)
            except Exception as exc:  # noqa: BLE001 - e.g. a GPU that hangs after resume
                log.warning("Local engine deep check failed: %s", exc)
                self._deep_due = False
                self._restart("inference hangs")
                self._set(DOWN, "inference hangs")
                return
        if self._attempts:
            log.info("Local engine healthy again, backoff reset")
        self._attempts, self._last_auto_start = 0, None
        self._set(READY, health=health, latency=latency)
