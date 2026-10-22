"""Local transcription engine: provider, catalog, packages, supervisor and the GUI-side controller."""
from __future__ import annotations

import copy
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from app.core.api_keys import (
    LOCAL_ENDPOINT,
    NO_KEY_PROFILE_ID,
    is_local_endpoint,
    provider_endpoint,
    provider_for_url,
    resolve_key_profile,
    selected_api_key,
    transcription_configured,
)
from app.core.constants import DEFAULT_CONFIG
from app.core.models import (
    LOCAL_TRANSCRIPTION_MODEL,
    TRANSCRIPTION_MODEL_OPTIONS,
    transcription_form_fields,
    transcription_reports_confidence,
)
from app.local_engine import availability, supervisor
from app.local_engine.catalog import (
    AUTO_DEVICE,
    COMPLETE_MARKER,
    CTRANSLATE2,
    OPENVINO,
    EngineSettings,
    mark_model_complete,
    model_downloaded,
    model_files_present,
    resolve_device,
    settings_from_config,
)
from app.local_engine.speech import is_hallucination
from app.local_engine.supervisor import OFF, READY, Supervisor

ROOT = Path(__file__).resolve().parents[1]


def _config(**overrides):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(overrides)
    return config


# --- Provider ---------------------------------------------------------------------------
def test_local_provider_needs_no_url_model_or_key():
    assert provider_for_url(LOCAL_ENDPOINT) == "local"
    assert provider_endpoint("local", "transcription") == LOCAL_ENDPOINT
    assert provider_endpoint("local", "rephrasing") == ""
    assert is_local_endpoint(LOCAL_ENDPOINT)  # never hedged
    config = _config(api_endpoint=LOCAL_ENDPOINT, api_key_profiles=[])
    assert resolve_key_profile(config, "transcription")["id"] == NO_KEY_PROFILE_ID
    assert selected_api_key(config, "transcription") == ""
    assert transcription_configured(config)


def test_local_model_has_no_dropdown_entry_and_no_quality_fallback():
    assert all(LOCAL_TRANSCRIPTION_MODEL not in models for models in TRANSCRIPTION_MODEL_OPTIONS.values())
    assert not transcription_reports_confidence(LOCAL_TRANSCRIPTION_MODEL)
    fields = transcription_form_fields(LOCAL_TRANSCRIPTION_MODEL, "Prompt", 0.3, "de", diagnostics=True)
    assert fields == {"model": LOCAL_TRANSCRIPTION_MODEL, "prompt": "Prompt", "language": "de"}


# --- Catalog ----------------------------------------------------------------------------
@pytest.mark.parametrize("available,expected", [
    (["cpu", "openvino_cpu", "intel_gpu", "cuda"], "cuda"),
    (["cpu", "openvino_cpu", "intel_gpu"], "intel_gpu"),
    (["cpu", "openvino_cpu"], "openvino_cpu"),
    (["cpu"], "cpu"),
    ([], None),
])
def test_auto_device_picks_the_fastest_present(available, expected):
    assert resolve_device(AUTO_DEVICE, available) == expected


def test_explicit_device_is_kept_even_when_not_detected():
    assert resolve_device("intel_npu", []) == "intel_npu"


def test_settings_from_config_use_the_backend_model_and_localhost_without_server(tmp_path):
    config = _config(local_server_host="0.0.0.0", local_models_dir=str(tmp_path))
    settings = settings_from_config(config, "cuda", str(tmp_path), 5555)
    assert settings.backend == CTRANSLATE2 and settings.model == config["local_model_ctranslate2"]
    assert settings.host == "127.0.0.1" and settings.endpoint == "http://127.0.0.1:5555/v1/audio/transcriptions"
    server = settings_from_config(_config(local_server_enabled=True, local_server_host="0.0.0.0"), "intel_gpu",
                                  str(tmp_path), 9017)
    assert server.backend == OPENVINO and server.host == "0.0.0.0"
    assert server.base_url == "http://127.0.0.1:9017"  # reached via loopback
    assert EngineSettings.from_json(server.to_json()) == server


def test_model_counts_as_downloaded_only_after_a_finished_download(tmp_path):
    assert not model_downloaded(CTRANSLATE2, tmp_path)
    (tmp_path / "model.bin").write_bytes(b"x")
    (tmp_path / "config.json").write_text("{}")
    assert model_files_present(CTRANSLATE2, tmp_path)
    assert not model_downloaded(CTRANSLATE2, tmp_path)  # interrupted download: no marker yet
    mark_model_complete(tmp_path)
    assert model_downloaded(CTRANSLATE2, tmp_path) and not model_downloaded(OPENVINO, tmp_path)
    (tmp_path / "model.bin").write_bytes(b"")  # empty weights never count
    assert not model_downloaded(CTRANSLATE2, tmp_path)
    mark_model_complete(tmp_path, complete=False)
    assert not (tmp_path / COMPLETE_MARKER).exists()


def test_failed_download_with_complete_files_loads_them_offline(tmp_path, monkeypatch):
    from app.local_engine import server

    settings = EngineSettings(device="cpu", model="org/model", models_dir=str(tmp_path), log_dir=str(tmp_path))
    folder = settings.model_path
    folder.mkdir()
    (folder / "model.bin").write_bytes(b"x")
    (folder / "config.json").write_text("{}")
    partial = folder / ".cache" / "huggingface" / "download" / "x.incomplete"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"12345")

    def offline(*_args, **_kwargs):
        raise OSError("network unreachable")

    monkeypatch.setattr(server, "download_model", offline)
    engine = server.Engine(settings)
    engine._download(folder)  # no exception: the files are there
    assert not partial.exists()  # the interrupted leftover is cleaned up
    (folder / "model.bin").unlink()
    with pytest.raises(server.DownloadError, match="network unreachable"):
        engine._download(folder)


def test_hallucinations_are_dropped_per_language():
    assert is_hallucination(" Vielen Dank. ", "de")
    assert not is_hallucination("Vielen Dank.", "en")
    assert is_hallucination("Thank you.", "")
    assert not is_hallucination("Vielen Dank für die schnelle Antwort.", "de")


# --- Packages ---------------------------------------------------------------------------
def test_pyproject_extras_match_the_installer_lists():
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"]
    assert extras["local"] == [package.requirement for package in availability.COMMON_PACKAGES]
    assert extras["local-intel"] == [package.requirement for package in availability.INTEL_PACKAGES]
    assert extras["local-nvidia"] == [package.requirement for package in availability.NVIDIA_PACKAGES]
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "faster-whisper" not in requirements and "openvino" not in requirements  # optional, never forced


@pytest.mark.parametrize("adapters,nvidia,intel", [
    (["NVIDIA GeForce RTX 3060 Laptop GPU", "Intel(R) Iris(R) Xe Graphics"], True, True),
    (["Intel(R) Arc(TM) Graphics"], False, True),
    (["AMD Radeon"], False, False),
])
def test_install_packages_follow_the_graphics_adapters(monkeypatch, adapters, nvidia, intel):
    monkeypatch.setattr(availability, "_installed", lambda _module: False)
    requirements = availability.install_packages(adapters)
    assert "faster-whisper>=1.1" in requirements
    assert any(req.startswith("nvidia-") for req in requirements) == nvidia
    assert any(req.startswith("openvino") for req in requirements) == intel


def test_nothing_to_install_when_everything_is_there(monkeypatch):
    monkeypatch.setattr(availability, "_installed", lambda _module: True)
    assert availability.install_packages(["NVIDIA RTX"]) == []


# --- Supervisor -------------------------------------------------------------------------
def test_backoff_grows_and_resets():
    steps = (0, 10, 30)
    assert supervisor.backoff_remaining(0, None, 100.0, steps) == 0
    assert supervisor.backoff_remaining(1, 100.0, 105.0, steps) == 5
    assert supervisor.backoff_remaining(5, 100.0, 200.0, steps) == 0


def test_engine_processes_are_recognized_by_their_module():
    assert supervisor.is_engine_process(["pythonw.exe", "-m", "app.local_engine.server"])
    assert not supervisor.is_engine_process(["python.exe", "-m", "whisperserver.server"])


@pytest.fixture
def fake_process(monkeypatch):
    """A supervisor whose engine process is simulated; ``health`` is what /health answers."""
    state = {"alive": False, "health": {"status": "ready", "pid": 1}, "started": 0}
    sup = Supervisor()

    def start(reason):
        state["alive"], state["started"] = True, state["started"] + 1
        sup._proc = Mock(pid=1)
        sup._started_at = time.monotonic()

    def stop():
        state["alive"] = False
        sup._proc = None

    monkeypatch.setattr(sup, "_start", start)
    monkeypatch.setattr(sup, "_stop", stop)
    monkeypatch.setattr(sup, "_alive", lambda: state["alive"])
    monkeypatch.setattr(supervisor, "get_health", lambda *_args, **kwargs: {**state["health"], "deep_s": 0.3})
    monkeypatch.setattr(supervisor, "_tree_pids", lambda pid: {pid})
    return sup, state


def _settings(**kw):
    return EngineSettings(device="cpu", model="m", models_dir="d", log_dir="l", **kw)


def test_supervisor_starts_checks_and_stops(fake_process):
    sup, state = fake_process
    sup.set_target(_settings())
    sup._handle_requests()
    sup.tick()
    assert state["started"] == 1 and sup.status.state == READY and sup.status.latency_s == 0.3
    sup.set_target(None, reason="stopped")
    sup._handle_requests()
    assert not state["alive"] and sup.status.state == OFF and sup.status.reason == "stopped"


def test_changed_settings_restart_the_engine(fake_process):
    sup, state = fake_process
    sup.set_target(_settings())
    sup._handle_requests()
    sup.set_target(_settings())  # unchanged: no restart
    sup._handle_requests()
    sup.set_target(_settings(vad=False))
    sup._handle_requests()
    assert state["started"] == 2


def test_idle_unload_ends_the_process(fake_process):
    sup, state = fake_process
    sup.idle_unload_s = 60
    sup.set_target(_settings())
    sup._handle_requests()
    sup._last_used -= 61
    sup.tick()
    assert not state["alive"] and sup.status.state == OFF and sup.status.reason == "idle" and not sup.wanted


def test_model_error_restarts_with_backoff(fake_process):
    sup, state = fake_process
    state["health"] = {"status": "error", "pid": 1, "error": "RuntimeError: no CUDA"}
    sup.set_target(_settings())
    sup._handle_requests()
    sup.tick()
    assert sup.status.state == "down" and "no CUDA" in sup.status.detail and state["started"] == 2
    sup.tick()  # the backoff holds the next restart
    assert state["started"] == 2


# --- Controller -------------------------------------------------------------------------
@pytest.fixture
def controller(monkeypatch, tmp_path):
    pytest.importorskip("PyQt6.QtWidgets")
    from app.audio.store import RecordingStore
    from app.context import AppContext
    from app.controllers import local_engine as module

    class Store:
        def __init__(self, config):
            self.config = config

        def load(self):
            return self.config, False

        def save(self, _config):
            return True

    ctx = AppContext(Store(_config(local_device="cpu")), RecordingStore(str(tmp_path)))
    engine = module.LocalEngineController(ctx)
    engine._supervisor = Mock(wanted=False, settings=None)
    monkeypatch.setattr(module.availability, "engine_installed", lambda: True)
    monkeypatch.setattr(module.availability, "backend_ready", lambda _backend: True)
    return engine, ctx.config, module


def test_disabled_engine_never_starts(controller):
    engine, config, _module = controller
    config.update(api_endpoint=LOCAL_ENDPOINT)
    engine.apply_config()
    engine.ensure_running()
    engine._supervisor.set_target.assert_not_called()
    assert engine.problem() == "local_engine_disabled"


def test_enabled_but_unused_engine_is_not_loaded(controller):
    engine, config, _module = controller
    config.update(local_engine_enabled=True)  # neither the "Local" provider nor the server
    engine.apply_config()
    engine._supervisor.set_target.assert_not_called()


@pytest.mark.parametrize("strategy,loads", [("preload", True), ("on_demand", False)])
def test_load_strategy(controller, strategy, loads):
    engine, config, _module = controller
    config.update(local_engine_enabled=True, api_endpoint=LOCAL_ENDPOINT, local_load_strategy=strategy)
    engine.apply_config()
    assert engine._supervisor.set_target.called == loads
    engine.ensure_running()  # a recording starts: on demand loads now
    assert engine._supervisor.set_target.called


def test_server_keeps_the_engine_loaded(controller):
    engine, config, _module = controller
    config.update(local_engine_enabled=True, local_server_enabled=True, local_load_strategy="on_demand",
                  local_idle_unload_minutes=5, local_server_port=9123)
    engine.apply_config()
    settings = engine._supervisor.set_target.call_args.args[0]
    assert settings.port == 9123 and engine._supervisor.idle_unload_s == 0


def test_macos_switches_the_engine_off(controller, monkeypatch):
    engine, config, module = controller
    monkeypatch.setattr(module, "LOCAL_ENGINE_SUPPORTED", False)
    config.update(local_engine_enabled=True, local_server_enabled=True)
    assert not engine.enabled() and not engine.supported()
    engine.apply_config()
    engine._supervisor.set_target.assert_not_called()


# --- Transcription request --------------------------------------------------------------
def test_unusable_engine_fails_the_request_with_its_reason(tmp_path):
    from app.core.timing import OperationTiming
    from app.services.transcription import TranscriptionError, TranscriptionRequest, transcribe

    request = TranscriptionRequest(api_key="", api_endpoint=LOCAL_ENDPOINT, audio_path=str(tmp_path / "a.wav"),
                                   prompt="", model=LOCAL_TRANSCRIPTION_MODEL, language="de", temperature=0,
                                   local_engine_error="Local transcription is turned off.")
    with pytest.raises(TranscriptionError, match="turned off"):
        transcribe(request, OperationTiming(), lambda key, **kw: key, on_compressing=Mock(), on_transcribing=Mock())


def test_waiting_for_a_downloading_engine_fails_at_once(monkeypatch):
    from app.services import transcription
    from app.services.transcription import TranscriptionError, TranscriptionRequest

    monkeypatch.setattr(transcription, "request", lambda *_a, **_kw: Mock(status_code=200,
                                                                          json=lambda: {"status": "downloading"}))
    request = TranscriptionRequest(api_key="", api_endpoint="http://127.0.0.1:1/v1/audio/transcriptions",
                                   audio_path="a.wav", prompt="", model=LOCAL_TRANSCRIPTION_MODEL, language="de",
                                   temperature=0, local_engine_wait_s=30)
    with pytest.raises(TranscriptionError, match="local_engine_downloading_error"):
        transcription._wait_for_local_engine(request, lambda key, **kw: key)


# --- Settings page ----------------------------------------------------------------------
@pytest.mark.parametrize("supported", [True, False])
def test_page_loads_an_enabled_engine_that_cannot_run(tmp_path, monkeypatch, supported):
    """Enabled in the config, but packages missing (or macOS): the window must still build."""
    pytest.importorskip("PyQt6.QtWidgets")
    import json

    from PyQt6.QtWidgets import QApplication

    from app.audio.store import RecordingStore
    from app.context import AppContext
    from app.controllers import local_engine as module
    from app.core.config_store import ConfigStore
    from app.core.hotkeys import normalize_hotkey_string
    from app.ui.settings.window import SettingsWindow

    qapp = QApplication.instance() or QApplication([])  # keep a reference: widgets need a live app
    monkeypatch.setattr(module, "LOCAL_ENGINE_SUPPORTED", supported)
    monkeypatch.setattr(module.availability, "engine_installed", lambda: False)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"ui_language": "en", "local_engine_enabled": True, "local_server_enabled": True,
                                "api_endpoint": LOCAL_ENDPOINT}), encoding="utf-8")
    ctx = AppContext(ConfigStore(str(path), normalize_hotkey_string), RecordingStore(str(tmp_path)))
    window = SettingsWindow(ctx, recording=Mock(selectable_input_devices=Mock(return_value=[])), hotkeys=Mock(),
                            files=Mock(log_file_exists=Mock(return_value=False)), quit_app=Mock())
    tab = window._local_engine_tab
    assert tab.enabled_checkbox.isChecked() == supported and tab.enabled_checkbox.isEnabled() == supported
    window.local_engine.apply_config()
    assert window.local_engine.status.state in ("off", module.UNAVAILABLE)
    window.local_engine.shutdown()
    window.deleteLater()
    qapp.processEvents()


def test_first_deep_check_runs_right_after_boot(fake_process, monkeypatch):
    """``time.monotonic`` starts near zero after boot; the first deep check must still run."""
    sup, _state = fake_process
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: 5.0)
    sup.set_target(_settings())
    sup._handle_requests()
    sup.tick()
    assert sup.status.latency_s == 0.3


def test_connection_test_blames_no_proxy_for_a_server_on_this_pc():
    """Nothing listens on a local port: that is not a proxy problem (local requests never use the proxy)."""
    httpx = pytest.importorskip("httpx")
    from app.services.net import classify_transport_error

    px = {"http": "http://127.0.0.1:3128", "https": "http://127.0.0.1:3128"}
    error = httpx.ConnectError("refused")
    assert classify_transport_error(error, "http://127.0.0.1:9017/v1/audio/transcriptions", px) == "local_down"
    assert classify_transport_error(error, "https://api.groq.com/openai/v1/audio/transcriptions", px) == "proxy"


def test_pip_skips_the_path_warnings():
    assert "--no-warn-script-location" in availability.pip_command(["x"])
