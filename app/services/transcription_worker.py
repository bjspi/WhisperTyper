"""Runs one transcription HTTP request off the GUI thread."""
from __future__ import annotations

import logging
import mimetypes
import os
import re
from typing import Any, Dict, Optional

import requests
from PyQt6.QtCore import QObject, pyqtSignal

from app.core import ffmpeg
from app.core.redaction import redact_for_log


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
                 min_bitrate_kbps: int = 80) -> None:
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
            proxies (Optional[Dict[str, str]]): Optional proxies mapping for requests (corporate proxy).
            ffmpeg_path (Optional[str]): Resolved ffmpeg binary. When set and ``audio_path`` is a
                video container, the audio track is extracted to a temp MP3 before upload. Also used
                to compress any file that exceeds ``max_upload_bytes``.
            max_upload_bytes (int): Upload-size ceiling for the endpoint. Files above it are
                compressed (mono/16 kHz, bitrate lowered as needed) to fit.
            min_bitrate_kbps (int): Floor for that compression; below this the file is rejected.
        """
        super().__init__()
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.audio_path = audio_path
        self.prompt = prompt
        self.model = re.sub(r"\s*\(.*?\)", "", model).strip()
        self.language = language.lower() if language else ""
        self.temperature = temperature
        self.proxies = proxies
        self.ffmpeg_path = ffmpeg_path
        self.max_upload_bytes = max_upload_bytes
        self.min_bitrate_kbps = min_bitrate_kbps

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
        # Path actually uploaded — may be a temp MP3 we extracted/compressed and must clean up.
        upload_path = self.audio_path
        extracted_temp: Optional[str] = None
        try:
            if not self.api_key:
                logging.debug("No API key provided in configuration.")
                raise ValueError("API key not found in configuration.")

            # Prepare the file for upload: videos get their audio extracted, and any file above the
            # endpoint's size limit is compressed to fit (mono/16 kHz, bitrate lowered as needed).
            # A video is only extractable when ffmpeg is present; that's guaranteed by the picker.
            is_video = ffmpeg.is_video_file(self.audio_path)
            # An oversized non-video file is compressed in-place below — announce that phase first so
            # the tray shows a "compressing…" spinner during the blocking re-encode. (Videos already
            # show the "extracting…" spinner raised by the caller.)
            if not is_video and self._is_oversized(self.audio_path):
                self.compressing.emit(os.path.basename(self.audio_path))
            upload_path, extracted_temp = ffmpeg.prepare_upload(
                self.ffmpeg_path, self.audio_path,
                transcode_source=bool(self.ffmpeg_path) and is_video,
                max_bytes=self.max_upload_bytes,
                min_bitrate_kbps=self.min_bitrate_kbps,
            )
            # Any pre-processing (extraction or compression) just finished — switch the spinner to
            # the transcription phase before uploading.
            if extracted_temp is not None:
                self.transcribing.emit()

            headers: Dict[str, str] = {"Authorization": f"Bearer {self.api_key}"}
            data: Dict[str, Any] = {"model": self.model, "prompt": self.prompt, "temperature": self.temperature}
            # Only add language if it's not empty (for auto-detection)
            if self.language:
                data["language"] = self.language

            log_data = dict(data)
            if "prompt" in log_data:
                log_data["prompt"] = redact_for_log(log_data["prompt"])
            logging.debug(f"API endpoint: {self.api_endpoint}")
            logging.debug(f"Request data: {log_data}")
            logging.debug(f"Audio file path: {upload_path}")

            with open(upload_path, 'rb') as audio_file:
                # Content type from the file extension so mp3/ogg/m4a uploads are labelled
                # correctly (recordings are WAV; user-picked files can be anything).
                content_type = mimetypes.guess_type(upload_path)[0] or "audio/wav"
                files = {"file": (os.path.basename(upload_path), audio_file, content_type)}
                # (connect, read) timeout. urllib3 applies the first value to the ENTIRE request
                # send — the TCP/TLS handshake *and* streaming the file body — not just connecting,
                # so it must budget the upload itself. A fixed few seconds trips "write operation
                # timed out" on multi-MB files; scale it to the payload assuming a pessimistic
                # ~64 KB/s uplink, with a floor for small recordings and a cap as a safety net. The
                # read timeout stays generous for the server's transcription of longer audio.
                try:
                    upload_size = os.path.getsize(upload_path)
                except OSError:
                    upload_size = 0
                send_timeout = min(600.0, max(30.0, upload_size / (64 * 1024)))
                logging.debug(
                    f"Sending POST request to API with file {files['file'][0]} "
                    f"({upload_size / (1024 * 1024):.1f} MB, send timeout {send_timeout:.0f}s)"
                )
                response = requests.post(
                    self.api_endpoint, headers=headers, files=files, data=data,
                    proxies=self.proxies, timeout=(send_timeout, 300)
                )

            logging.debug(f"API response status: {response.status_code}")
            if response.status_code == 200:
                transcription: str = response.json().get("text", "")
                logging.info(f"Transcription result: {redact_for_log(transcription)}")
                self.finished.emit(transcription)
            else:
                error_msg = f"API Error: {response.status_code}\n{response.text}"
                logging.error(error_msg)
                self.error.emit(error_msg, self.audio_path)
        except Exception as e:
            error_msg = f"An unexpected error occurred in worker:\n{str(e)}"
            logging.error(error_msg)
            self.error.emit(error_msg, self.audio_path)
        finally:
            # Remove the temp MP3 we extracted from a video (the original file is untouched).
            if extracted_temp:
                try:
                    os.remove(extracted_temp)
                except OSError as cleanup_err:
                    logging.debug(f"Could not remove extracted temp file: {cleanup_err}")
