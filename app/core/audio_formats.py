"""Which media files the transcription APIs accept, and how a picked file is uploaded.

OpenAI and Groq accept flac, m4a, mp3, mp4, mpeg, mpga, ogg, wav and webm, judged by the file
name. Voice messages such as WhatsApp ``.opus`` or Telegram ``.oga`` are Ogg containers under a
rejected extension; ``.aac`` (raw ADTS) is not accepted at all. With FFmpeg those are converted;
without it the Ogg variants are uploaded under an ``.ogg`` name. Content types come from a fixed
table, because OS MIME databases disagree (or know nothing) about several of these extensions.
"""
from __future__ import annotations

import mimetypes
import os

#: Extensions the transcription APIs accept as uploaded.
ACCEPTED_AUDIO_EXTENSIONS = frozenset({
    ".flac", ".m4a", ".mp3", ".mp4", ".mpeg", ".mpga", ".ogg", ".wav", ".webm",
})
#: Audio the APIs reject by extension; converted with FFmpeg when it is available.
NORMALIZED_AUDIO_EXTENSIONS = frozenset({".opus", ".oga", ".aac"})
#: Everything offered as audio in the file picker.
AUDIO_EXTENSIONS = ACCEPTED_AUDIO_EXTENSIONS | NORMALIZED_AUDIO_EXTENSIONS
#: Containers treated as video: their audio track is extracted first when FFmpeg is available.
VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".wmv", ".flv", ".mpg", ".mpeg", ".ts", ".3gp",
})

# Ogg variants the APIs accept once they carry the .ogg extension (same container, no conversion).
_UPLOAD_EXTENSION_ALIASES = {".opus": ".ogg", ".oga": ".ogg"}
_CONTENT_TYPES = {
    ".aac": "audio/aac", ".flac": "audio/flac", ".m4a": "audio/mp4", ".mp3": "audio/mpeg",
    ".mp4": "audio/mp4", ".mpeg": "audio/mpeg", ".mpga": "audio/mpeg", ".oga": "audio/ogg",
    ".ogg": "audio/ogg", ".opus": "audio/ogg", ".wav": "audio/wav", ".webm": "audio/webm",
}


def _extension(path: str) -> str:
    """Lowercased extension including the dot."""
    return os.path.splitext(path)[1].lower()


def is_video_file(path: str) -> bool:
    """True for a known video container whose audio track FFmpeg should extract."""
    return _extension(path) in VIDEO_EXTENSIONS


def needs_audio_normalization(path: str) -> bool:
    """True for audio the APIs reject by extension and FFmpeg should convert."""
    return _extension(path) in NORMALIZED_AUDIO_EXTENSIONS


def requires_ffmpeg(path: str) -> bool:
    """True if the file cannot be uploaded at all without FFmpeg (a video the APIs reject)."""
    extension = _extension(path)
    return extension in VIDEO_EXTENSIONS and extension not in ACCEPTED_AUDIO_EXTENSIONS


def upload_filename(path: str) -> str:
    """The name sent in the multipart upload; Ogg variants travel as ``.ogg``."""
    stem, extension = os.path.splitext(os.path.basename(path))
    return stem + _UPLOAD_EXTENSION_ALIASES.get(extension.lower(), extension)


def upload_content_type(path: str) -> str:
    """Content type of the upload, independent of the operating system's MIME database."""
    return (_CONTENT_TYPES.get(_extension(path))
            or mimetypes.guess_type(path)[0] or "application/octet-stream")
