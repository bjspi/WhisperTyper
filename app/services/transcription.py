"""One transcription request: prepare the upload (format, compression, extraction) and send it.

Runs on a worker thread; progress phases and user-facing errors are reported through the
caller's callbacks and translation lookup, so this module needs no Qt.
"""
from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

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
    with open(upload.path, 'rb') as audio_file:
        files = {"file": (audio_formats.upload_filename(upload.path), audio_file, content_type)}
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

    logging.debug(f"API response status: {response.status_code}")
    if response.status_code != 200:
        raise TranscriptionError(tr("transcription_api_error", status=response.status_code, details=response.text))
    transcription: str = response.json().get("text", "")
    timing.mark("transcription_response_parsed")
    logging.info(f"Transcription result: {redact_for_log(transcription)}")
    return transcription
