"""Which engine packages are installed, which to install, and which graphics adapters exist.

Cheap enough for the GUI process: packages are looked up with ``find_spec`` (nothing heavy is
imported) and the adapters are read from the operating system.
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from typing import List, NamedTuple, Tuple

from app.local_engine.catalog import CTRANSLATE2, OPENVINO


class Package(NamedTuple):
    """A pip requirement and the module that proves it is installed."""

    requirement: str
    module: str


#: Needed by every device: the server, audio decoding, silence removal (faster-whisper's Silero VAD),
#: the model download and the CTranslate2 backend (CPU).
COMMON_PACKAGES: Tuple[Package, ...] = (
    Package("faster-whisper>=1.1", "faster_whisper"),
    # Silero VAD; versions before 1.19 do not load with numpy 2.
    Package("onnxruntime>=1.19", "onnxruntime"),
    Package("fastapi>=0.110", "fastapi"),
    Package("uvicorn>=0.29", "uvicorn"),
    Package("python-multipart>=0.0.9", "multipart"),
    Package("huggingface_hub>=0.23", "huggingface_hub"),
    Package("psutil>=5.9", "psutil"),
    Package("numpy", "numpy"),
    Package("av>=12", "av"),
    Package("truststore>=0.9", "truststore"),
)
#: Intel GPU / NPU (and the OpenVINO CPU device).
INTEL_PACKAGES: Tuple[Package, ...] = (Package("openvino-genai>=2025.3", "openvino_genai"),)
#: CUDA libraries for CTranslate2 on NVIDIA GPUs (about 1.5 GB).
NVIDIA_PACKAGES: Tuple[Package, ...] = (
    Package("nvidia-cublas-cu12", "nvidia.cublas"),
    Package("nvidia-cudnn-cu12>=9,<10", "nvidia.cudnn"),
)


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def missing(packages: Tuple[Package, ...]) -> List[Package]:
    """The packages of ``packages`` that are not installed."""
    importlib.invalidate_caches()  # a pip install in this session must be seen
    return [package for package in packages if not _installed(package.module)]


def packages_for_backend(backend: str) -> Tuple[Package, ...]:
    """What a backend needs to run (without the optional CUDA libraries)."""
    return COMMON_PACKAGES + (INTEL_PACKAGES if backend == OPENVINO else ())


def engine_installed() -> bool:
    """The common packages are there, so the engine can at least run on the CPU."""
    return not missing(COMMON_PACKAGES)


def display_adapters() -> List[str]:
    """Names of the graphics adapters (Windows; elsewhere an empty list)."""
    if sys.platform != "win32":
        return []
    import ctypes
    from ctypes import wintypes

    class DisplayDevice(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("DeviceName", wintypes.WCHAR * 32),
                    ("DeviceString", wintypes.WCHAR * 128), ("StateFlags", wintypes.DWORD),
                    ("DeviceID", wintypes.WCHAR * 128), ("DeviceKey", wintypes.WCHAR * 128)]

    names: List[str] = []
    device = DisplayDevice()
    device.cb = ctypes.sizeof(device)
    index = 0
    try:
        while ctypes.windll.user32.EnumDisplayDevicesW(None, index, ctypes.byref(device), 0):
            if device.DeviceString and device.DeviceString not in names:
                names.append(device.DeviceString)
            index += 1
    except (AttributeError, OSError) as exc:
        logging.debug("display_adapters failed: %s", exc)
    return names


def install_packages(adapters: List[str]) -> List[str]:
    """The pip requirements to install for this computer's hardware (missing ones only)."""
    wanted = COMMON_PACKAGES
    names = " ".join(adapters).lower()
    if "nvidia" in names:
        wanted += NVIDIA_PACKAGES
    if "intel" in names:
        wanted += INTEL_PACKAGES
    return [package.requirement for package in missing(wanted)]


def console_python() -> str:
    """This environment's interpreter with a console: under ``pythonw`` child processes get no output pipes."""
    import os

    folder, name = os.path.split(sys.executable)
    if name.lower() == "pythonw.exe":
        candidate = os.path.join(folder, "python.exe")
        if os.path.isfile(candidate):
            return candidate
    return sys.executable


def pip_command(requirements: List[str]) -> List[str]:
    """Arguments of a pip run that installs ``requirements`` into this interpreter's environment."""
    return ["-m", "pip", "install", "--disable-pip-version-check", "--no-warn-script-location", *requirements]


def backend_ready(backend: str) -> bool:
    """The packages of ``backend`` are installed."""
    if backend == CTRANSLATE2:
        return engine_installed()
    return not missing(packages_for_backend(backend))
