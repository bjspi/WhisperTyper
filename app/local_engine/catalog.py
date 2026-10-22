"""Devices, backends and models of the local engine, and the settings the engine process runs with.

Two backends cover the common hardware:

* ``openvino``: OpenVINO GenAI on Intel GPUs (Arc, Iris Xe, Core Ultra), Intel NPUs and any CPU.
  OpenVINO's GPU plugin only drives Intel GPUs.
* ``ctranslate2``: faster-whisper / CTranslate2 on NVIDIA GPUs (CUDA) and any CPU.
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple

OPENVINO, CTRANSLATE2 = "openvino", "ctranslate2"

#: Device choices (config value -> backend, the backend's device name). "auto" is resolved by
#: :func:`resolve_device` from the detected hardware.
DEVICES: Dict[str, Tuple[str, str]] = {
    "cuda": (CTRANSLATE2, "cuda"),
    "intel_gpu": (OPENVINO, "GPU"),
    "intel_npu": (OPENVINO, "NPU"),
    "openvino_cpu": (OPENVINO, "CPU"),
    "cpu": (CTRANSLATE2, "cpu"),
}
AUTO_DEVICE = "auto"
#: Preference of "auto": the fastest device that is present.
_AUTO_ORDER = ("cuda", "intel_gpu", "openvino_cpu", "cpu")


class LocalModel(NamedTuple):
    """A selectable model: Hugging Face repository, display name, approximate download size."""

    repo: str
    name: str
    size_mb: int
    recommended: bool = False


MODELS: Dict[str, Tuple[LocalModel, ...]] = {
    OPENVINO: (
        LocalModel("OpenVINO/whisper-large-v3-turbo-int4-ov", "large-v3-turbo int4", 550, recommended=True),
        LocalModel("OpenVINO/whisper-large-v3-turbo-int8-ov", "large-v3-turbo int8", 900),
        LocalModel("OpenVINO/whisper-large-v3-turbo-fp16-ov", "large-v3-turbo fp16", 1650),
        LocalModel("OpenVINO/whisper-large-v3-int8-ov", "large-v3 int8", 1600),
    ),
    CTRANSLATE2: (
        LocalModel("mobiuslabsgmbh/faster-whisper-large-v3-turbo", "large-v3-turbo", 1620, recommended=True),
        LocalModel("Systran/faster-whisper-large-v3", "large-v3", 3090),
        LocalModel("Systran/faster-whisper-medium", "medium", 1530),
        LocalModel("Systran/faster-whisper-small", "small", 480),
    ),
}
#: Config key holding the chosen model of each backend, so switching the device keeps both choices.
MODEL_CONFIG_KEYS = {OPENVINO: "local_model_openvino", CTRANSLATE2: "local_model_ctranslate2"}
DEFAULT_MODELS = {backend: models[0].repo for backend, models in MODELS.items()}

LOAD_STRATEGIES = ("preload", "on_demand")
DEFAULT_SERVER_PORT = 9017
#: The engine process reads its settings (JSON) from this environment variable.
SETTINGS_ENV = "WHISPERTYPER_ENGINE_SETTINGS"


def default_models_dir(app_data_dir: str) -> Path:
    """Models live in the app data folder unless the user picked another one."""
    return Path(app_data_dir) / "local_models"


def model_dir_name(repo: str) -> str:
    """Folder of a downloaded model: the repository name without its owner."""
    return repo.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def model_path(models_dir: Path, repo: str) -> Path:
    """Where ``repo`` is (or will be) stored; an absolute path is a model folder of its own."""
    candidate = Path(repo)
    return candidate if candidate.is_absolute() else models_dir / model_dir_name(repo)


def model_downloaded(backend: str, path: Path) -> bool:
    """Whether ``path`` holds a complete model for ``backend``."""
    if backend == OPENVINO:
        return (path / "openvino_encoder_model.xml").is_file() and (path / "openvino_decoder_model.xml").is_file()
    return (path / "model.bin").is_file() and (path / "config.json").is_file()


def resolve_device(choice: str, available: List[str]) -> Optional[str]:
    """The concrete device for a config choice: "auto" picks the best of ``available``.

    Returns None when "auto" finds nothing usable. An explicit choice is kept even when it was
    not detected; the engine then reports why it cannot load.
    """
    if choice in DEVICES:
        return choice
    return next((device for device in _AUTO_ORDER if device in available), None)


def backend_of(device: str) -> str:
    """The backend that drives ``device``."""
    return DEVICES[device][0]


@dataclasses.dataclass(frozen=True)
class EngineSettings:
    """Everything the engine process needs; serialized into :data:`SETTINGS_ENV`."""

    device: str
    model: str
    models_dir: str
    log_dir: str
    host: str = "127.0.0.1"
    port: int = DEFAULT_SERVER_PORT
    vad: bool = True
    #: Log at DEBUG level (the app's "debug logging" setting).
    debug: bool = False
    #: Keep transcribed text out of the log (the app's redaction setting).
    redact: bool = True

    @property
    def backend(self) -> str:
        """The backend of the device."""
        return backend_of(self.device)

    @property
    def model_path(self) -> Path:
        """Folder of the model."""
        return model_path(Path(self.models_dir), self.model)

    @property
    def base_url(self) -> str:
        """URL clients on this computer use; a server listening on all interfaces is reached via loopback."""
        host = "127.0.0.1" if self.host in ("0.0.0.0", "", "::") else self.host
        if ":" in host:
            host = f"[{host}]"
        return f"http://{host}:{self.port}"

    @property
    def endpoint(self) -> str:
        """The OpenAI-compatible transcription URL."""
        return f"{self.base_url}/v1/audio/transcriptions"

    def to_json(self) -> str:
        """Serialize for the engine process."""
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_json(cls, text: str) -> "EngineSettings":
        """Read settings written by ``to_json`` (unknown keys are ignored)."""
        data: Dict[str, Any] = json.loads(text)
        fields = {field.name for field in dataclasses.fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in fields})


def settings_from_config(config: Mapping[str, Any], device: str, app_data_dir: str, port: int) -> EngineSettings:
    """Engine settings for ``device`` from the app config; ``port`` is the one to listen on."""
    backend = backend_of(device)
    model = str(config.get(MODEL_CONFIG_KEYS[backend]) or DEFAULT_MODELS[backend]).strip()
    models_dir = str(config.get("local_models_dir") or "").strip() or str(default_models_dir(app_data_dir))
    server = bool(config.get("local_server_enabled"))
    host = str(config.get("local_server_host") or "127.0.0.1").strip() if server else "127.0.0.1"
    return EngineSettings(device=device, model=model, models_dir=os.path.expanduser(models_dir),
                          log_dir=app_data_dir, host=host, port=port, vad=bool(config.get("local_vad", True)),
                          debug=bool(config.get("debug_logging", False)),
                          redact=bool(config.get("redact_transcription_in_log", True)))
