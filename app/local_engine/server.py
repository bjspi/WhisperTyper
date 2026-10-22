"""The engine process: loads the model and serves an OpenAI-compatible transcription endpoint.

    POST /v1/audio/transcriptions   multipart: file, [language], [prompt], [response_format]
    GET  /v1/models
    GET  /health                    {"status": "downloading" | "loading" | "ready" | "error", ...}
    GET  /health?deep=1             additionally runs a tiny inference (detects a hung GPU)

Started and supervised by ``app.local_engine.supervisor``; the settings come as JSON in the
``WHISPERTYPER_ENGINE_SETTINGS`` environment variable. Run directly for debugging:
``python -m app.local_engine.server`` (with that variable set).
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from app.local_engine.backends import Backend, create_backend
from app.local_engine.catalog import (
    SETTINGS_ENV,
    EngineSettings,
    mark_model_complete,
    model_downloaded,
    model_files_present,
)
from app.local_engine.speech import MIN_SPEECH_SAMPLES, SAMPLE_RATE, decode_audio, is_hallucination, strip_silence

try:  # module level: FastAPI resolves the (string) annotations of the endpoints in this namespace
    from fastapi import UploadFile
except ImportError:  # pragma: no cover - the supervisor checks the packages before starting us
    UploadFile = None  # type: ignore[assignment,misc]

log = logging.getLogger("whispertyper.engine")


def _folder_size(path: Path) -> int:
    """Bytes below ``path`` (including partial downloads)."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def remove_partial_downloads(path: Path) -> int:
    """Delete ``*.incomplete`` leftovers of interrupted downloads (they are not resumed); returns the bytes freed."""
    freed = 0
    for leftover in (path / ".cache" / "huggingface" / "download").glob("**/*.incomplete"):
        try:
            size = leftover.stat().st_size
            leftover.unlink()
            freed += size
        except OSError:
            pass
    return freed


def ensure_models_dir(models_dir: Path) -> None:
    """Create the models folder with a ``.nobackup`` marker (backup tools may skip downloads)."""
    models_dir.mkdir(parents=True, exist_ok=True)
    marker = models_dir / ".nobackup"
    if not marker.exists():
        marker.write_text("Downloaded Whisper models - backup tools may skip this folder.\n", encoding="utf-8")


def download_model(settings: EngineSettings, on_total: Any = None) -> None:
    """Download the settings' model from Hugging Face into the models folder."""
    path = settings.model_path
    ensure_models_dir(Path(settings.models_dir))
    try:
        import truststore

        truststore.inject_into_ssl()  # corporate proxies: trust the system certificate store
    except ImportError:
        pass
    from huggingface_hub import HfApi, snapshot_download

    if on_total is not None:
        try:
            info = HfApi().model_info(settings.model, files_metadata=True)
            on_total(sum(sibling.size or 0 for sibling in info.siblings or []))
        except Exception as exc:  # noqa: BLE001 - the size is only for the progress display
            log.debug("Model size lookup failed: %s", exc)
    snapshot_download(settings.model, local_dir=str(path))


class DownloadError(RuntimeError):
    """The model could not be downloaded (shown as is; the supervisor retries with backoff)."""


def _megabytes_text(size: int) -> str:
    return f"{size / (1024 * 1024):.0f} MB"


class Engine:
    """Owns the backend; loads it in the background so /health answers right away."""

    def __init__(self, settings: EngineSettings) -> None:
        """The model loads later, on its own thread (``load``)."""
        self.settings = settings
        self.status = "loading"
        self.error = ""
        self.backend: Optional[Backend] = None
        self.started = time.time()
        self.requests = 0
        self.download_total = 0

    def load(self) -> None:
        """Download the model if needed, then load it; a failure sets ``status`` to error."""
        backend_name = self.settings.backend
        path = self.settings.model_path
        try:
            if not model_downloaded(backend_name, path):
                self._download(path)
            self.status = "loading"
            t0 = time.perf_counter()
            backend = create_backend(self.settings)
            log.info("Loading %s with %s on %s (%s)", path, type(backend).__name__, self.settings.device,
                     self.settings.backend)
            backend.load(path, Path(self.settings.models_dir) / ".ovcache")
            self.backend = backend
            self.status = "ready"
            log.info("Model %s on %s ready in %.1f s", path.name, self.settings.device, time.perf_counter() - t0)
        except DownloadError as exc:
            self.status, self.error = "error", str(exc)
            log.exception("Downloading the model failed")
        except Exception as exc:  # noqa: BLE001 - reported via /health; the supervisor restarts us
            self.status, self.error = "error", f"{type(exc).__name__}: {exc}"
            log.exception("Loading the model failed")
            if self.settings.model_path.is_dir() and not Path(self.settings.model).is_absolute():
                # Damaged files: the next start verifies the download again and fetches what is missing.
                mark_model_complete(path, complete=False)
                log.info("Completion marker removed from %s; the next start checks the download again", path)

    def _download(self, path: Path) -> None:
        """Download (or resume, or verify) the model; a usable set of files survives being offline."""
        self.status = "downloading"
        partial = path.is_dir() and any(path.iterdir())
        log.info("%s %s into %s", "Completing/verifying" if partial else "Downloading", self.settings.model, path)
        freed = remove_partial_downloads(path)
        if freed:
            log.info("Removed %s of an interrupted download", _megabytes_text(freed))
        t0 = time.perf_counter()
        try:
            download_model(self.settings, on_total=lambda total: setattr(self, "download_total", total))
        except Exception as exc:  # noqa: BLE001 - offline, proxy, rate limit, disk full …
            if model_files_present(self.settings.backend, path):
                log.warning("Download check failed (%s); the model files are there, loading them", exc)
                return
            raise DownloadError(f"Download failed: {type(exc).__name__}: {exc}") from exc
        if not model_files_present(self.settings.backend, path):
            raise DownloadError(f"{self.settings.model} is not a {self.settings.backend} Whisper model")
        mark_model_complete(path)
        remove_partial_downloads(path)
        log.info("Download of %s complete in %.0f s (%s)", self.settings.model, time.perf_counter() - t0,
                 _megabytes_text(_folder_size(path)))

    def health(self) -> Dict[str, Any]:
        """The /health answer: status, model, device, uptime, requests, download progress."""
        body: Dict[str, Any] = {
            "status": self.status, "model": self.settings.model_path.name, "device": self.settings.device,
            "backend": self.settings.backend, "pid": os.getpid(), "uptime_s": round(time.time() - self.started),
            "requests": self.requests,
        }
        if self.error:
            body["error"] = self.error
        if self.status == "downloading":
            done = _folder_size(self.settings.model_path)
            body["downloaded_bytes"] = min(done, self.download_total) if self.download_total else done
            body["download_total_bytes"] = self.download_total
        return body

    def transcribe(self, data: bytes, language: str = "", prompt: Optional[str] = None) -> Tuple[str, float]:
        """Text and audio duration (seconds) of an uploaded file."""
        assert self.backend is not None
        t0 = time.perf_counter()
        audio = decode_audio(data)
        duration = len(audio) / SAMPLE_RATE
        t_decoded = time.perf_counter()
        speech = self._speech(audio)
        t_vad = time.perf_counter()
        self.requests += 1
        timings = (f"decode={1000 * (t_decoded - t0):.0f}ms vad={1000 * (t_vad - t_decoded):.0f}ms "
                   f"audio={duration:.2f}s speech={len(speech) / SAMPLE_RATE:.2f}s")
        if len(speech) < MIN_SPEECH_SAMPLES:
            log.info("request #%d: no speech found (%s, %d bytes)", self.requests, timings, len(data))
            return "", duration
        text = self.backend.generate(speech, language, prompt)
        t_done = time.perf_counter()
        hallucination = is_hallucination(text, language)
        log.info("request #%d: language=%s prompt_chars=%d %s generate=%.0fms total=%.0fms chars=%d%s",
                 self.requests, language or "auto", len(prompt or ""), timings, 1000 * (t_done - t_vad),
                 1000 * (t_done - t0), len(text), " dropped=hallucination" if hallucination else "")
        if not self.settings.redact:
            log.debug("request #%d text: %r", self.requests, text)
        return ("" if hallucination else text), duration

    def _speech(self, audio: Any) -> Any:
        """The audio without pauses; unchanged when silence removal is off or cannot run."""
        if not self.settings.vad:
            return audio
        try:
            return strip_silence(audio)
        except Exception as exc:  # noqa: BLE001 - e.g. a broken onnxruntime; transcribe all of it
            log.warning("Silence removal failed, transcribing without it: %s", exc)
            return audio

    def deep_check(self) -> float:
        """Seconds a tiny inference takes; raises when the device fails."""
        import numpy as np

        assert self.backend is not None
        t0 = time.perf_counter()
        noise = (np.random.default_rng(0).standard_normal(SAMPLE_RATE) * 0.001).astype(np.float32)
        self.backend.generate(noise, "en", None)
        seconds = time.perf_counter() - t0
        log.info("Deep check: inference %.3f s", seconds)
        return seconds


def create_app(engine: Engine) -> Any:
    """The FastAPI application with the health, models and transcription routes."""
    from fastapi import FastAPI, File, Form, HTTPException
    from fastapi.concurrency import run_in_threadpool
    from fastapi.responses import PlainTextResponse

    app = FastAPI(title="WhisperTyper local transcription", version="1")

    @app.get("/health")
    def health(deep: int = 0) -> Dict[str, Any]:
        body = engine.health()
        if deep and engine.status == "ready":
            body["deep_s"] = round(engine.deep_check(), 3)
        return body

    @app.get("/v1/models")
    @app.get("/openai/v1/models")
    def models() -> Dict[str, Any]:
        model_id = engine.settings.model_path.name
        return {"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "local"}]}

    @app.post("/v1/audio/transcriptions")
    @app.post("/openai/v1/audio/transcriptions")  # Groq-style path
    async def transcriptions(file: UploadFile = File(...), model: str = Form(""),
                             language: str = Form(""), prompt: Optional[str] = Form(None),
                             response_format: str = Form("json"), temperature: float = Form(0.0)) -> Any:
        if engine.status != "ready":
            raise HTTPException(503, f"model not ready ({engine.status})")
        data = await file.read()
        if not data:
            raise HTTPException(400, "empty file")
        language = language.strip().lower().split("-")[0]
        try:
            text, duration = await run_in_threadpool(engine.transcribe, data, language, prompt)
        except Exception as exc:  # noqa: BLE001 - e.g. undecodable audio
            log.warning("Transcription of %s (%d bytes) failed", file.filename, len(data), exc_info=True)
            raise HTTPException(400, f"transcription failed: {exc}") from exc
        if response_format in ("text", "srt", "vtt"):
            return PlainTextResponse(text)
        if response_format == "verbose_json":
            segment = {"id": 0, "seek": 0, "start": 0.0, "end": duration, "text": text, "tokens": [],
                       "temperature": 0.0, "avg_logprob": 0.0, "compression_ratio": 1.0, "no_speech_prob": 0.0}
            return {"task": "transcribe", "language": language, "duration": duration, "text": text,
                    "segments": [segment] if text else [], "words": []}
        return {"text": text}

    return app


def log_environment(settings: EngineSettings) -> None:
    """Versions and devices for a support log: what this engine runs on."""
    import platform
    from importlib import metadata

    log.info("Python %s (%s) on %s %s", platform.python_version(), sys.executable, platform.system(),
             platform.version())
    versions = []
    for package in ("openvino", "openvino-genai", "ctranslate2", "faster-whisper", "onnxruntime", "numpy", "av",
                    "fastapi", "uvicorn", "huggingface_hub", "nvidia-cublas-cu12", "nvidia-cudnn-cu12"):
        try:
            versions.append(f"{package}={metadata.version(package)}")
        except metadata.PackageNotFoundError:
            versions.append(f"{package}=-")
    log.info("Packages: %s", " ".join(versions))
    log.info("Settings: %s", settings.to_json())
    try:
        from app.local_engine.devices import detect

        log.info("Devices: %s", detect())
    except Exception:  # noqa: BLE001 - diagnostics only
        log.exception("Device detection failed")


def setup_logging(settings: EngineSettings) -> None:
    """Log to <app data>/local_engine.log (rotating); stray output goes to a stderr log."""
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    sys.excepthook = lambda kind, value, tb: log.critical("Unhandled exception", exc_info=(kind, value, tb))
    log_file = log_dir / "local_engine.log"
    if sys.stderr is None:  # pythonw: keep stray prints and tracebacks
        stream = open(log_file.with_suffix(".stderr.log"), "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = stream
    handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if settings.debug else logging.INFO)
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "multipart", "python_multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> int:
    """Run the engine until the supervisor stops the process."""
    settings = EngineSettings.from_json(os.environ[SETTINGS_ENV])
    setup_logging(settings)
    log.info("Engine starting (PID %d) on %s:%d, device %s, model %s", os.getpid(), settings.host, settings.port,
             settings.device, settings.model)
    log_environment(settings)
    import uvicorn

    engine = Engine(settings)
    threading.Thread(target=engine.load, name="model-load", daemon=True).start()
    uvicorn.run(create_app(engine), host=settings.host, port=settings.port, log_config=None, access_log=False)
    log.info("Engine stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
