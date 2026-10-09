"""Runs one transcription HTTP request off the GUI thread."""
from __future__ import annotations

import logging
import mimetypes
import os
import tempfile
from typing import Callable, Dict, Optional

from PyQt6.QtCore import QObject, pyqtSignal

from app.audio.aac_encoder import available_aac_bitrates, encode_wav_to_aac
from app.core import ffmpeg
from app.core.models import transcription_form_fields
from app.core.redaction import redact_for_log
from app.core.textutil import clean_model_name
from app.core.timing import OperationTiming
from app.services.http_transport import request


class TranscriptionWorker(QObject):
    """
    Runs the API request in a separate thread to avoid blocking the GUI.
    """

    finished = pyqtSignal(str)
    error = pyqtSignal(str, str)
    # Emitted once pre-processing (video extraction / compression) finishes and the upload begins,
    # so the tray spinner can switch to the "transcribing…" phase.
    transcribing = pyqtSignal()
    # Emitted just before an oversized file is compressed, so the tray can show a distinct
    # "compressing…" spinner during the (blocking) re-encode before the transcription phase.
    compressing = pyqtSignal(str)

    def __init__(self, api_key: str, api_endpoint: str, audio_path: str, prompt: str, model: str,
                 language: str, temperature: float, proxies: Optional[Dict[str, str]] = None,
                 ffmpeg_path: Optional[str] = None, max_upload_bytes: int = 24 * 1024 * 1024,
                 min_bitrate_kbps: int = 80, timing: Optional[OperationTiming] = None,
                 recording_format: Optional[str] = None, recording_bitrate_kbps: int = 64,
                 ffmpeg_setting: str = "", tr: Optional[Callable[..., str]] = None) -> None:
        """
        Initializes the transcription worker.

        Args:
            api_key (str): The API key for the transcription service.
            api_endpoint (str): The URL of the transcription API endpoint.
            audio_path (str): The local path to the audio file to be transcribed.
            prompt (str): A prompt to guide the transcription model.
            model (str): The name of the transcription model to use.
            language (str): The language of the audio in ISO 639-1 format (e.g., "en", "de"). Can be empty for auto-detection.
            temperature (float): The sampling temperature for the model.
            proxies (Optional[Dict[str, str]]): Optional {"http": url, "https": url} proxy mapping (corporate proxy).
            ffmpeg_path (Optional[str]): Resolved ffmpeg binary. When set and ``audio_path`` is a
                video container, the audio track is extracted to a temp MP3 before upload. Also used
                to compress any file that exceeds ``max_upload_bytes``.
            max_upload_bytes (int): Upload-size ceiling for the endpoint. Files above it are
                compressed (mono/16 kHz, bitrate lowered as needed) to fit.
            min_bitrate_kbps (int): Floor for that compression; below this the file is rejected.
            timing: This operation's timing state, shared with the GUI result callbacks.
            recording_format: WAV or AAC (PyAV) for microphone recordings; None keeps file/video processing.
            recording_bitrate_kbps: AAC bitrate; WAV remains uncompressed PCM.
            ffmpeg_setting: Configured ffmpeg path, resolved only when a recording exceeds the
                upload limit, so ordinary recordings never pay for the lookup.
            tr: Translation lookup for user-facing errors (a plain dict read, safe off the GUI thread).
        """
        super().__init__()
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.audio_path = audio_path
        self.prompt = prompt
        self.model = clean_model_name(model)
        self.language = language.lower() if language else ""
        self.temperature = temperature
        self.proxies = proxies
        self.ffmpeg_path = ffmpeg_path
        self.max_upload_bytes = max_upload_bytes
        self.min_bitrate_kbps = min_bitrate_kbps
        self.timing = timing or OperationTiming()
        self.recording_format = recording_format
        self.recording_bitrate_kbps = recording_bitrate_kbps
        self.ffmpeg_setting = ffmpeg_setting
        self.tr: Callable[..., str] = tr or (lambda key, **_kwargs: key)

    def _compress_oversized_recording(self, encoded_temp: Optional[str]) -> tuple[str, Optional[str], int]:
        """Shrink a long recording with ffmpeg, as for picked files; only reached above the limit."""
        ffmpeg_path = ffmpeg.resolve_ffmpeg(self.ffmpeg_setting)
        if not ffmpeg_path:
            raise ValueError(self.tr("recording_upload_too_large"))
        if encoded_temp:
            # Re-encode from the retained WAV rather than transcoding the AAC a second time.
            os.remove(encoded_temp)
        self.compressing.emit(os.path.basename(self.audio_path))
        with self.timing.span("audio_compress"):
            upload_path, temp = ffmpeg.prepare_upload(
                ffmpeg_path, self.audio_path, transcode_source=False,
                max_bytes=self.max_upload_bytes, min_bitrate_kbps=self.min_bitrate_kbps,
            )
        upload_size = os.path.getsize(upload_path)
        logging.info("recording_upload op=%s compressed=ffmpeg upload_bytes=%s", self.timing.operation_id, upload_size)
        return upload_path, temp, upload_size

    def _is_oversized(self, path: str) -> bool:
        """True if ``path`` is larger than the upload limit (and would therefore be compressed)."""
        try:
            return os.path.getsize(path) > self.max_upload_bytes
        except OSError:
            return False

    def run(self) -> None:
        """
        Executes the transcription request and emits the corresponding signal.
        """
        logging.info("TranscriptionWorker started.")
        self.timing.mark("transcription_worker_start")
        # Path actually uploaded — may be a temporary AAC or extracted/compressed file.
        upload_path = self.audio_path
        extracted_temp: Optional[str] = None
        upload_size = 0
        try:
            if not self.api_key:
                logging.debug("No API key provided in configuration.")
                raise ValueError(self.tr("transcription_api_key_missing"))

            self.timing.mark("upload_prepare_start")

            if self.recording_format is not None:
                if self.recording_format not in ("wav", "aac"):
                    raise ValueError(self.tr("recording_format_unsupported"))
                source_size = os.path.getsize(self.audio_path)
                if source_size == 0:
                    raise ValueError(self.tr("recording_file_empty"))
                recording_format = self.recording_format
                if recording_format == "aac" and self.recording_bitrate_kbps not in available_aac_bitrates():
                    # PyAV is optional (e.g. a macOS build without it): degrade to WAV instead of failing.
                    logging.warning("recording_upload op=%s aac_unavailable=true; uploading WAV", self.timing.operation_id)
                    recording_format = "wav"
                if recording_format == "aac":
                    descriptor, extracted_temp = tempfile.mkstemp(prefix="whispertyper_upload_", suffix=".m4a")
                    os.close(descriptor)
                    with self.timing.span("audio_encode"):
                        encode_wav_to_aac(self.audio_path, extracted_temp, self.recording_bitrate_kbps)
                    upload_path = extracted_temp
                upload_size = os.path.getsize(upload_path) if extracted_temp else source_size
                if upload_size == 0:
                    raise ValueError(self.tr("recording_encode_empty"))
                logging.info("recording_upload op=%s format=%s bitrate_kbps=%s source_bytes=%s upload_bytes=%s",
                             self.timing.operation_id, recording_format,
                             self.recording_bitrate_kbps if recording_format == "aac" else "pcm",
                             source_size, upload_size)
                if upload_size > self.max_upload_bytes:
                    upload_path, extracted_temp, upload_size = self._compress_oversized_recording(extracted_temp)
            else:
                # Picked files retain the existing video extraction / oversized-file compression.
                is_video = ffmpeg.is_video_file(self.audio_path)
                if not is_video and self._is_oversized(self.audio_path):
                    self.compressing.emit(os.path.basename(self.audio_path))
                upload_path, extracted_temp = ffmpeg.prepare_upload(
                    self.ffmpeg_path, self.audio_path,
                    transcode_source=bool(self.ffmpeg_path) and is_video,
                    max_bytes=self.max_upload_bytes,
                    min_bitrate_kbps=self.min_bitrate_kbps,
                )
            self.timing.mark("upload_prepare_end")
            # Any pre-processing (extraction or compression) just finished — switch the spinner to
            # the transcription phase before uploading.
            if extracted_temp is not None:
                self.transcribing.emit()

            headers: Dict[str, str] = {"Authorization": f"Bearer {self.api_key}"}
            data = transcription_form_fields(self.model, self.prompt, self.temperature, self.language)

            log_data = dict(data)
            if "prompt" in log_data:
                log_data["prompt"] = redact_for_log(log_data["prompt"])
            logging.debug(f"API endpoint: {self.api_endpoint}")
            logging.debug(f"Request data: {log_data}")
            logging.debug(f"Audio file path: {upload_path}")

            with open(upload_path, 'rb') as audio_file:
                # M4A is consistently labelled across OS MIME databases.
                content_type = "audio/mp4" if upload_path.lower().endswith(".m4a") else mimetypes.guess_type(upload_path)[0] or "audio/wav"
                files = {"file": (os.path.basename(upload_path), audio_file, content_type)}
                # (send, read) timeout. The transport applies the first value to connecting (TCP/TLS)
                # and to uploading the body, so it budgets the upload: scaled to the payload at a
                # pessimistic ~64 KB/s uplink, with a floor for small recordings and a cap as a
                # safety net. The read timeout stays generous for transcribing longer audio.
                if self.recording_format is None:
                    try:
                        upload_size = os.path.getsize(upload_path)
                    except OSError:
                        upload_size = 0
                send_timeout = min(600.0, max(30.0, upload_size / (64 * 1024)))
                logging.debug(
                    f"Sending POST request to API with file {files['file'][0]} "
                    f"({upload_size / (1024 * 1024):.1f} MB, send timeout {send_timeout:.0f}s)"
                )
                self.timing.mark("transcription_request_start")
                response = request(
                    "POST", self.api_endpoint, timing=self.timing, stage="transcription", headers=headers, files=files, data=data,
                    proxies=self.proxies, timeout=(send_timeout, 300)
                )
                # The transport returns only after reading the complete response body.
                self.timing.mark("transcription_response_received")

            logging.debug(f"API response status: {response.status_code}")
            if response.status_code == 200:
                transcription: str = response.json().get("text", "")
                self.timing.mark("transcription_response_parsed")
                logging.info(f"Transcription result: {redact_for_log(transcription)}")
                self.timing.mark("transcription_worker_ready")
                self.finished.emit(transcription)
            else:
                error_msg = self.tr("transcription_api_error", status=response.status_code, details=response.text)
                logging.error(error_msg)
                self.timing.mark("transcription_failed")
                self.error.emit(error_msg, self.audio_path)
        except Exception as e:
            error_msg = self.tr("transcription_worker_error", error=e)
            logging.error(error_msg)
            self.timing.mark("transcription_failed")
            self.error.emit(error_msg, self.audio_path)
        finally:
            # Remove only the temporary upload; retain the original recording/file for retry.
            if extracted_temp:
                try:
                    os.remove(extracted_temp)
                except OSError as cleanup_err:
                    logging.debug(f"Could not remove extracted temp file: {cleanup_err}")
