"""One transcription request: prepare the upload (format, compression, extraction) and send it.

Runs on a worker thread; progress phases and user-facing errors are reported through the
caller's callbacks and translation lookup, so this module needs no Qt.
"""
from __future__ import annotations

import logging
import os
import queue
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import httpx

from app.audio.aac_encoder import available_aac_bitrates, encode_wav_to_aac
from app.core import audio_formats
from app.core.models import transcription_form_fields
from app.core.redaction import redact_for_log
from app.core.textutil import clean_model_name
from app.core.timing import OperationTiming
from app.services import ffmpeg
from app.services.http_transport import request

Translate = Callable[..., str]

#: Recording formats a retained microphone WAV can be uploaded as.
RECORDING_FORMATS = ("wav", "aac")
#: Upload budget for the send timeout: a pessimistic ~64 KB/s uplink, 30 s floor, 10 min cap.
_UPLINK_BYTES_PER_S = 64 * 1024
_SEND_TIMEOUT_RANGE_S = (30.0, 600.0)
#: The server may need a while to transcribe long audio.
_READ_TIMEOUT_S = 300.0

# Hedging thresholds. "Upload finished" means the body was handed to the OS send buffer, so a
# small upload counts as finished at once and a stall then shows up as a missing response.
#: A second attempt starts when the upload is not finished after this long, or after the time
#: the upload size needs at the given rate (long recordings) ...
_HEDGE_UPLOAD_MIN_S = 0.5
_HEDGE_UPLOAD_BYTES_PER_S = 1.5 * 1024 * 1024
#: ... or when no response byte arrives within this long after the upload finished (plus some
#: time per uploaded MB, because the server transcribes longer audio for longer).
_HEDGE_RESPONSE_MIN_S = 1.0
_HEDGE_RESPONSE_S_PER_MB = 0.5


class TranscriptionError(Exception):
    """A failure whose (already translated) message is shown to the user as-is."""


@dataclass(frozen=True)
class TranscriptionRequest:
    """Everything one request needs, snapshotted on the GUI thread.

    ``recording_format`` is set for retained microphone recordings ("wav"/"aac"); picked
    files leave it None and keep video extraction and oversized-file compression, which
    use the resolved ``ffmpeg_path``. Recordings resolve ``ffmpeg_setting`` only when they
    exceed ``max_upload_bytes``, so ordinary recordings never pay for the lookup.
    """

    api_key: str
    api_endpoint: str
    audio_path: str
    prompt: str
    model: str
    language: str
    temperature: float
    proxies: Optional[Dict[str, str]] = None
    ffmpeg_path: Optional[str] = None
    max_upload_bytes: int = 24 * 1024 * 1024
    min_bitrate_kbps: int = 80
    recording_format: Optional[str] = None
    recording_bitrate_kbps: int = 64
    ffmpeg_setting: str = ""
    #: Key for a parallel second attempt over a fresh connection when the first one stalls
    #: (the next Groq key when rotation is on, else the same key); None disables hedging.
    hedge_api_key: Optional[str] = None


@dataclass(frozen=True)
class _Upload:
    """The file that is actually sent; ``temp`` is deleted afterwards (the source never is)."""

    path: str
    temp: Optional[str]
    size: int


def transcribe(req: TranscriptionRequest, timing: OperationTiming, tr: Translate, *,
               on_compressing: Callable[[str], None], on_transcribing: Callable[[], None]) -> str:
    """Upload ``req.audio_path`` and return the transcription text.

    Raises:
        TranscriptionError: For conditions with a translated, user-facing explanation.
    """
    if not req.api_key:
        raise TranscriptionError(tr("transcription_api_key_missing"))
    upload: Optional[_Upload] = None
    try:
        with timing.span("upload_prepare"):
            upload = (_prepare_recording(req, timing, tr, on_compressing) if req.recording_format is not None
                      else _prepare_file(req, on_compressing))
        # Pre-processing (extraction or compression) just finished: the upload phase begins.
        if upload.temp is not None:
            on_transcribing()
        return _send(req, upload, timing, tr)
    finally:
        # Remove only the temporary upload; the original recording/file stays for retries.
        if upload is not None and upload.temp:
            _remove(upload.temp)


def _remove(path: str) -> None:
    """Best-effort deletion of a temporary upload."""
    try:
        os.remove(path)
    except OSError as cleanup_err:
        logging.debug(f"Could not remove extracted temp file: {cleanup_err}")


def _prepare_recording(req: TranscriptionRequest, timing: OperationTiming, tr: Translate,
                       on_compressing: Callable[[str], None]) -> _Upload:
    """Upload a retained WAV as-is or as AAC; compress it with ffmpeg only above the limit."""
    if req.recording_format not in RECORDING_FORMATS:
        raise TranscriptionError(tr("recording_format_unsupported"))
    source_size = os.path.getsize(req.audio_path)
    if source_size == 0:
        raise TranscriptionError(tr("recording_file_empty"))
    recording_format = req.recording_format
    if recording_format == "aac" and req.recording_bitrate_kbps not in available_aac_bitrates():
        # PyAV is optional (e.g. a macOS build without it): degrade to WAV instead of failing.
        logging.warning("recording_upload op=%s aac_unavailable=true; uploading WAV", timing.operation_id)
        recording_format = "wav"

    upload = _Upload(req.audio_path, None, source_size)
    if recording_format == "aac":
        descriptor, temp = tempfile.mkstemp(prefix="whispertyper_upload_", suffix=".m4a")
        os.close(descriptor)
        try:
            with timing.span("audio_encode"):
                encode_wav_to_aac(req.audio_path, temp, req.recording_bitrate_kbps)
            upload = _Upload(temp, temp, os.path.getsize(temp))
        except BaseException:
            _remove(temp)
            raise
    if upload.size == 0:
        if upload.temp:
            _remove(upload.temp)
        raise TranscriptionError(tr("recording_encode_empty"))
    logging.info("recording_upload op=%s format=%s bitrate_kbps=%s source_bytes=%s upload_bytes=%s",
                 timing.operation_id, recording_format,
                 req.recording_bitrate_kbps if recording_format == "aac" else "pcm", source_size, upload.size)
    if upload.size <= req.max_upload_bytes:
        return upload

    # Long recording: shrink it like a picked file, from the retained WAV (not the AAC).
    ffmpeg_path = ffmpeg.resolve_ffmpeg(req.ffmpeg_setting)
    if upload.temp:
        _remove(upload.temp)
    if not ffmpeg_path:
        raise TranscriptionError(tr("recording_upload_too_large"))
    on_compressing(os.path.basename(req.audio_path))
    with timing.span("audio_compress"):
        compressed, compressed_temp = ffmpeg.prepare_upload(
            ffmpeg_path, req.audio_path, transcode_source=False,
            max_bytes=req.max_upload_bytes, min_bitrate_kbps=req.min_bitrate_kbps)
    size = os.path.getsize(compressed)
    logging.info("recording_upload op=%s compressed=ffmpeg upload_bytes=%s", timing.operation_id, size)
    return _Upload(compressed, compressed_temp, size)


def _prepare_file(req: TranscriptionRequest, on_compressing: Callable[[str], None]) -> _Upload:
    """Picked files: extract a video's audio, convert rejected audio formats, compress above the limit."""
    transcode = bool(req.ffmpeg_path) and (audio_formats.is_video_file(req.audio_path)
                                           or audio_formats.needs_audio_normalization(req.audio_path))
    if not transcode and _file_size(req.audio_path) > req.max_upload_bytes:
        on_compressing(os.path.basename(req.audio_path))
    path, temp = ffmpeg.prepare_upload(
        req.ffmpeg_path, req.audio_path,
        transcode_source=transcode,
        max_bytes=req.max_upload_bytes, min_bitrate_kbps=req.min_bitrate_kbps,
    )
    return _Upload(path, temp, _file_size(path))


def _file_size(path: str) -> int:
    """Size in bytes, or 0 when the file cannot be read."""
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _timeouts(upload_size: int) -> Tuple[float, float]:
    """(send, read) timeouts; the send budget covers connecting and uploading the body."""
    low, high = _SEND_TIMEOUT_RANGE_S
    return min(high, max(low, upload_size / _UPLINK_BYTES_PER_S)), _READ_TIMEOUT_S


def _send(req: TranscriptionRequest, upload: _Upload, timing: OperationTiming, tr: Translate) -> str:
    """POST the multipart request and return the transcription text."""
    model = clean_model_name(req.model)
    data = transcription_form_fields(model, req.prompt, req.temperature, req.language.lower() if req.language else "")
    log_data = dict(data)
    if "prompt" in log_data:
        log_data["prompt"] = redact_for_log(log_data["prompt"])
    logging.debug(f"API endpoint: {req.api_endpoint}")
    logging.debug(f"Request data: {log_data}")
    logging.debug(f"Audio file path: {upload.path}")

    send_timeout, read_timeout = _timeouts(upload.size)
    content_type = audio_formats.upload_content_type(upload.path)
    filename = audio_formats.upload_filename(upload.path)
    if req.hedge_api_key is not None:
        return _send_hedged(req, upload, timing, tr, data, filename, content_type, (send_timeout, read_timeout))
    with open(upload.path, 'rb') as audio_file:
        files = {"file": (filename, audio_file, content_type)}
        logging.debug(f"Sending POST request to API with file {files['file'][0]} "
                      f"({upload.size / (1024 * 1024):.1f} MB, send timeout {send_timeout:.0f}s)")
        timing.mark("transcription_request_start")
        response = request(
            "POST", req.api_endpoint, timing=timing, stage="transcription",
            headers={"Authorization": f"Bearer {req.api_key}"}, files=files, data=data,
            proxies=req.proxies, timeout=(send_timeout, read_timeout),
        )
        # The transport returns only after reading the complete response body.
        timing.mark("transcription_response_received")

    transcription = _parse(response, tr)
    timing.mark("transcription_response_parsed")
    logging.info(f"Transcription result: {redact_for_log(transcription)}")
    return transcription


def _parse(response: httpx.Response, tr: Translate) -> str:
    """The transcription text of a complete response; any non-200 status is a user-facing error."""
    logging.debug(f"API response status: {response.status_code}")
    if response.status_code != 200:
        raise TranscriptionError(tr("transcription_api_error", status=response.status_code, details=response.text))
    text: str = response.json().get("text", "")
    return text


def hedge_thresholds(upload_bytes: int) -> Tuple[float, float]:
    """(upload budget, response wait after the upload) before a second attempt starts, in seconds."""
    megabytes = upload_bytes / (1024 * 1024)
    return (max(_HEDGE_UPLOAD_MIN_S, upload_bytes / _HEDGE_UPLOAD_BYTES_PER_S),
            _HEDGE_RESPONSE_MIN_S + megabytes * _HEDGE_RESPONSE_S_PER_MB)


def _send_hedged(req: TranscriptionRequest, upload: _Upload, timing: OperationTiming, tr: Translate,
                 data: Dict[str, object], filename: str, content_type: str,
                 timeouts: Tuple[float, float]) -> str:
    """Send like ``_send``; if the upload or the response stalls, race a second attempt.

    The second attempt uses ``req.hedge_api_key`` over a fresh connection, so a pooled connection
    stuck in TCP retransmission backoff cannot hold it up. The first successful response wins;
    the other attempt finishes in the background and is discarded. A failure that arrives before
    any stall is reported as without hedging.
    """
    with open(upload.path, 'rb') as audio_file:
        body = audio_file.read()  # Each attempt sends this buffer; no file handle is shared.
    results: "queue.SimpleQueue[Tuple[int, Optional[str], Optional[Exception]]]" = queue.SimpleQueue()
    uploaded, answered, hedged = threading.Event(), threading.Event(), threading.Event()
    uploaded_at: List[float] = []
    finished_ms: Dict[int, float] = {}

    def watch(phase: str) -> None:
        """Progress of the first attempt, reported from its transport thread."""
        if phase == "request_sent" and not uploaded.is_set():
            uploaded_at.append(time.monotonic())
            uploaded.set()
        elif phase == "first_byte":
            answered.set()

    def attempt(index: int, api_key: str, stage: str, fresh: bool) -> None:
        """One POST; its outcome goes to the coordinator instead of being raised here."""
        try:
            response = request(
                "POST", req.api_endpoint, timing=timing, stage=stage,
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (filename, body, content_type)}, data=data,
                proxies=req.proxies, timeout=timeouts, fresh_connection=fresh,
                on_phase=watch if index == 1 else None,
            )
            outcome: Tuple[int, Optional[str], Optional[Exception]] = (index, _parse(response, tr), None)
        except Exception as error:
            outcome = (index, None, error)
        finished_ms[index] = (time.monotonic() - started) * 1000
        if hedged.is_set():
            # Every attempt of a hedged request reports its own end, the discarded one included.
            logging.info("transcription_hedge op=%s attempt=%s finished_ms=%.0f outcome=%s", timing.operation_id,
                         index, finished_ms[index], "ok" if outcome[2] is None else type(outcome[2]).__name__)
        results.put(outcome)

    def launch(index: int, api_key: str, stage: str, fresh: bool) -> None:
        threading.Thread(target=attempt, args=(index, api_key, stage, fresh),
                         name=f"TranscriptionAttempt{index}", daemon=True).start()

    upload_budget, response_wait = hedge_thresholds(upload.size)
    timing.mark("transcription_request_start")
    started = time.monotonic()
    launch(1, req.api_key, "transcription", False)
    trigger = ""
    while not trigger:
        if answered.is_set():
            remaining: Optional[float] = None  # The response is arriving: nothing to hedge against.
        elif uploaded.is_set():
            remaining = max(0.0, uploaded_at[0] + response_wait - time.monotonic())
        else:
            remaining = max(0.0, started + upload_budget - time.monotonic())
        try:
            _index, text, error = results.get(timeout=remaining)
        except queue.Empty:
            if not answered.is_set():
                trigger = "response" if uploaded.is_set() else "upload"
            continue
        return _finish(timing, text, error, winner=1, trigger="")

    hedged.set()
    timing.mark("transcription_hedge_started")
    logging.info("transcription_hedge op=%s started trigger=%s after_ms=%.0f upload_bytes=%s fresh_connection=true",
                 timing.operation_id, trigger, (time.monotonic() - started) * 1000, upload.size)
    launch(2, req.hedge_api_key or req.api_key, "transcription_hedge", True)
    errors: List[Exception] = []
    for _ in range(2):
        index, text, error = results.get()
        if error is None:
            return _finish(timing, text, None, winner=index, trigger=trigger, winner_ms=finished_ms.get(index))
        errors.append(error)
    logging.info("transcription_hedge op=%s trigger=%s winner=none", timing.operation_id, trigger)
    raise errors[0]


def _finish(timing: OperationTiming, text: Optional[str], error: Optional[Exception],
            winner: int, trigger: str, winner_ms: Optional[float] = None) -> str:
    """Complete the operation with the winning attempt's outcome (``trigger`` empty: no hedge ran)."""
    timing.mark("transcription_response_received")
    if trigger:
        timing.mark(f"transcription_hedge_won_{winner}")
        logging.info("transcription_hedge op=%s trigger=%s winner=%s winner_ms=%s", timing.operation_id, trigger,
                     winner, "?" if winner_ms is None else f"{winner_ms:.0f}")
    if error is not None:
        raise error
    transcription = text or ""
    timing.mark("transcription_response_parsed")
    logging.info(f"Transcription result: {redact_for_log(transcription)}")
    return transcription
