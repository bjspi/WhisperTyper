"""Inference backends of the engine process: OpenVINO GenAI and CTranslate2 (faster-whisper).

Imported by the engine process only; the heavy packages are imported when a backend loads.
"""
from __future__ import annotations

import glob
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Optional

from app.local_engine.catalog import CTRANSLATE2, DEVICES, EngineSettings

SAMPLE_RATE = 16000
log = logging.getLogger("whispertyper.engine")


def add_nvidia_dll_dirs() -> None:
    """Make the CUDA libraries of the ``nvidia-*-cu12`` wheels loadable (Windows).

    CTranslate2 loads cuBLAS and cuDNN by name at runtime; the wheels put the DLLs into
    ``site-packages/nvidia/<library>/bin``, which is on no search path by default.
    """
    if sys.platform != "win32":
        return
    try:
        import nvidia  # namespace package of the wheels
    except ImportError:
        return
    folders = [folder for root in getattr(nvidia, "__path__", [])
               for folder in glob.glob(os.path.join(root, "*", "bin")) if os.path.isdir(folder)]
    for folder in folders:
        try:
            os.add_dll_directory(folder)
        except OSError:
            pass
    os.environ["PATH"] = os.pathsep.join(folders + [os.environ.get("PATH", "")])
    log.debug("CUDA library folders: %s", folders)


class Backend:
    """Loads one model on one device and transcribes 16 kHz mono float32 audio."""

    def __init__(self, settings: EngineSettings) -> None:
        """Prepare the backend; ``load`` does the work."""
        self.settings = settings
        self._lock = threading.Lock()

    def load(self, path: Path, cache_dir: Path) -> None:
        """Load the model at ``path`` (``cache_dir`` holds compiled kernels)."""
        raise NotImplementedError

    def generate(self, audio: Any, language: str, prompt: Optional[str]) -> str:
        """Text of ``audio``; an empty ``language`` detects it, ``prompt`` guides spelling."""
        raise NotImplementedError


class OpenVinoBackend(Backend):
    """OpenVINO GenAI ``WhisperPipeline`` (Intel GPU, NPU, CPU)."""

    def load(self, path: Path, cache_dir: Path) -> None:
        """Compile the pipeline for the device and run it once."""
        import numpy as np
        import openvino_genai as ov_genai

        cache_dir.mkdir(parents=True, exist_ok=True)
        device = DEVICES[self.settings.device][1]
        log.info("OpenVINO GenAI %s: WhisperPipeline on %s, kernel cache %s",
                 getattr(ov_genai, "__version__", "?"), device, cache_dir)
        # Compiled GPU kernels are cached, so later loads (reboot, wake, model switch) are quick.
        self._pipe = ov_genai.WhisperPipeline(str(path), device, CACHE_DIR=str(cache_dir))
        self._config = self._pipe.get_generation_config()
        self._config.task = "transcribe"
        self._config.return_timestamps = False
        self._pipe.generate(np.zeros(SAMPLE_RATE, dtype=np.float32).tolist(), self._config)  # compile now

    def generate(self, audio: Any, language: str, prompt: Optional[str]) -> str:
        """Transcribe with the language and prompt of this request."""
        with self._lock:
            config = self._config
            config.language = f"<|{language}|>" if language else None
            use_prompt = bool(prompt) and hasattr(config, "initial_prompt")
            if use_prompt:
                config.initial_prompt = prompt
            try:
                return str(self._pipe.generate(audio.tolist(), config)).strip()
            finally:
                if use_prompt:
                    config.initial_prompt = None


class CTranslate2Backend(Backend):
    """faster-whisper ``WhisperModel`` (NVIDIA CUDA in float16, CPU in int8)."""

    def load(self, path: Path, cache_dir: Path) -> None:
        """Load the model (float16 on CUDA, int8 on the CPU) and run it once."""
        import numpy as np

        device = DEVICES[self.settings.device][1]
        if device == "cuda":
            add_nvidia_dll_dirs()
        from faster_whisper import WhisperModel

        compute_type = "float16" if device == "cuda" else "int8"
        import ctranslate2

        log.info("CTranslate2 %s: device %s (CUDA devices: %d), compute type %s", ctranslate2.__version__, device,
                 ctranslate2.get_cuda_device_count(), compute_type)
        self._model = WhisperModel(str(path), device=device, compute_type=compute_type)
        self.generate(np.zeros(SAMPLE_RATE, dtype=np.float32), "en", None)  # load the kernels now

    def generate(self, audio: Any, language: str, prompt: Optional[str]) -> str:
        """Transcribe in one pass (beam search 5, no timestamps)."""
        with self._lock:
            segments, _info = self._model.transcribe(
                audio, language=language or None, initial_prompt=prompt or None, beam_size=5,
                vad_filter=False, condition_on_previous_text=False, without_timestamps=True)
            return "".join(segment.text for segment in segments).strip()


def create_backend(settings: EngineSettings) -> Backend:
    """The backend for the settings' device."""
    return CTranslate2Backend(settings) if settings.backend == CTRANSLATE2 else OpenVinoBackend(settings)
