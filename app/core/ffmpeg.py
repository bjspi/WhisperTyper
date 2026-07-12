"""FFmpeg detection, video→audio extraction, and upload-size compression.

Single responsibility: locate a usable ``ffmpeg`` binary and turn whatever file the user picked
into an MP3 the transcription API will accept — extracting audio from video containers and, when
a file is above the endpoint's upload limit, shrinking it enough to fit. No Qt, no app state —
callers pass the configured path plus limits and get back plain values / a temp file path.

Whisper-style endpoints reject video containers (or silently misbehave on them), and even
audio-capable containers can be too large to upload. We re-encode to mono, 16 kHz MP3 — Whisper
resamples to 16 kHz mono internally anyway, so this costs no transcription quality while shrinking
the file dramatically. The bitrate stays at a comfortable default and is only lowered (down to a
configurable floor) when the file would otherwise exceed the upload limit.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from typing import List, Optional, Tuple

from app.core.env import is_WINDOWS, no_window_kwargs

# Containers we treat as "video" — picking one requires ffmpeg to extract the audio track first.
# Kept lowercase; compared against the file's lowercased extension.
VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".wmv", ".flv", ".mpg", ".mpeg", ".ts", ".3gp",
})

# Encode target: mono, 16 kHz MP3. Whisper works at 16 kHz mono internally, so this is lossless for
# transcription while keeping files small. The default bitrate is used unless the file must shrink
# further to fit under the upload limit.
_TARGET_SAMPLE_RATE = 16000
_DEFAULT_BITRATE_KBPS = 128
# Aim a little below the hard limit so container/MP3 framing overhead can't push us over it.
_SIZE_SAFETY = 0.95


class AudioTooLargeError(RuntimeError):
    """The file can't be brought under the upload limit (too long even at the minimum bitrate)."""


def is_video_file(path: str) -> bool:
    """True if ``path``'s extension is a known video container that needs audio extraction."""
    return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def resolve_ffmpeg(configured_path: str = "") -> Optional[str]:
    """Return a usable ffmpeg executable path, or None if none is available.

    Preference order: an explicitly configured path (if it points at a real, runnable file),
    then whatever ``ffmpeg`` resolves to on PATH. The configured path may be either the binary
    itself or a directory that contains it.
    """
    configured = (configured_path or "").strip().strip('"')
    if configured:
        candidate = configured
        if os.path.isdir(candidate):
            # Allow pointing at the folder that holds ffmpeg(.exe).
            exe = "ffmpeg.exe" if is_WINDOWS else "ffmpeg"
            candidate = os.path.join(candidate, exe)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
        # Fall back to PATH lookup of the configured value as a bare command name.
        from shutil import which
        found = which(configured)
        if found:
            return found

    from shutil import which
    return which("ffmpeg")


def probe_version(exe: Optional[str]) -> Optional[str]:
    """Run ``<exe> -version`` and return a short version string (e.g. "6.1.1"), or None on failure."""
    if not exe:
        return None
    try:
        result = subprocess.run(
            [exe, "-version"],
            capture_output=True, text=True, timeout=10, **no_window_kwargs(),
        )
    except (OSError, subprocess.SubprocessError) as e:
        logging.debug(f"ffmpeg version probe failed for {exe!r}: {e}")
        return None
    if result.returncode != 0:
        return None
    # First line looks like: "ffmpeg version 6.1.1-full_build-www.gyan.dev Copyright ..."
    first_line = (result.stdout or "").splitlines()[0] if result.stdout else ""
    parts = first_line.split()
    if len(parts) >= 3 and parts[0] == "ffmpeg" and parts[1] == "version":
        return parts[2]
    return first_line.strip() or "unknown"


def probe_duration(exe: str, src_path: str) -> Optional[float]:
    """Return ``src_path``'s duration in seconds, or None if ffmpeg can't report it.

    Parsed from ffmpeg's own stderr (``ffmpeg -i <file>`` prints a ``Duration: HH:MM:SS.ss`` line),
    so it needs no separate ffprobe binary. Used to pick a bitrate that keeps the encode under the
    upload limit.
    """
    try:
        result = subprocess.run(
            [exe, "-i", src_path],
            capture_output=True, text=True, timeout=60, **no_window_kwargs(),
        )
    except (OSError, subprocess.SubprocessError) as e:
        logging.debug(f"ffmpeg duration probe failed for {src_path!r}: {e}")
        return None
    # ffmpeg exits non-zero here (no output file given) but still prints the Duration line.
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr or "")
    if not match:
        return None
    hours, minutes, seconds = int(match.group(1)), int(match.group(2)), float(match.group(3))
    return hours * 3600 + minutes * 60 + seconds


def _encode_audio(exe: str, src_path: str, dst_path: str, bitrate_kbps: int) -> None:
    """Re-encode ``src_path`` to a mono, 16 kHz MP3 at ``bitrate_kbps`` (dropping any video track).

    Raises:
        RuntimeError: If ffmpeg exits non-zero or produces no output.
    """
    cmd: List[str] = [
        exe, "-y",                          # overwrite the temp file we just created
        "-i", src_path,
        "-vn",                              # drop any video stream
        "-ac", "1",                         # downmix to mono (Whisper works in mono)
        "-ar", str(_TARGET_SAMPLE_RATE),    # 16 kHz — Whisper's internal rate
        "-acodec", "libmp3lame",
        "-b:a", f"{bitrate_kbps}k",
        dst_path,
    ]
    logging.info(
        f"Encoding audio via ffmpeg: {os.path.basename(src_path)} -> "
        f"{os.path.basename(dst_path)} (mono 16 kHz {bitrate_kbps} kbps)"
    )
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, **no_window_kwargs(),
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise RuntimeError(f"ffmpeg could not be run: {e}")

    if result.returncode != 0:
        # ffmpeg writes diagnostics to stderr; surface the tail so failures are debuggable.
        tail = (result.stderr or "").strip().splitlines()[-3:]
        raise RuntimeError("ffmpeg failed to encode audio:\n" + "\n".join(tail))
    if not os.path.isfile(dst_path) or os.path.getsize(dst_path) == 0:
        raise RuntimeError("ffmpeg produced no audio output (the file may have no audio track).")


def prepare_upload(
    exe: Optional[str],
    src_path: str,
    *,
    transcode_source: bool,
    max_bytes: int,
    min_bitrate_kbps: int,
    default_bitrate_kbps: int = _DEFAULT_BITRATE_KBPS,
) -> Tuple[str, Optional[str]]:
    """Return a path to upload plus an optional temp file the caller must delete.

    Produces a file the endpoint will accept:
      * ``transcode_source`` (a video container) is always re-encoded to extract its audio.
      * Any file larger than ``max_bytes`` is compressed to fit.
      * Everything else is uploaded untouched (returned path == ``src_path``, temp == None).

    When re-encoding, the bitrate stays at ``default_bitrate_kbps`` and is only lowered — never
    below ``min_bitrate_kbps`` — when the default wouldn't fit under ``max_bytes``.

    Returns:
        (upload_path, temp_path). ``temp_path`` is None when the original file is uploaded as-is;
        otherwise it equals ``upload_path`` and the caller must remove it once done.

    Raises:
        AudioTooLargeError: The file can't be shrunk under ``max_bytes`` (too long even at the floor),
            or it's oversized and no ffmpeg is available to compress it.
        RuntimeError: If ffmpeg fails while encoding.
    """
    try:
        src_size: Optional[int] = os.path.getsize(src_path)
    except OSError:
        src_size = None

    oversized = src_size is not None and src_size > max_bytes
    if not transcode_source and not oversized:
        return src_path, None

    if not exe:
        # Only reachable for an oversized audio file with no ffmpeg (video is guarded upstream).
        size_mb = (src_size or 0) / (1024 * 1024)
        limit_mb = max_bytes / (1024 * 1024)
        raise AudioTooLargeError(
            f"Audio file is {size_mb:.1f} MB, above the {limit_mb:.0f} MB upload limit, and no "
            f"ffmpeg is available to compress it. Set an ffmpeg path in settings or use a smaller file."
        )

    duration = probe_duration(exe, src_path)
    if duration and duration > 0:
        # Bitrate (kbps) whose encoded size lands right at the safety-adjusted limit.
        fitting_kbps = (max_bytes * _SIZE_SAFETY * 8) / duration / 1000
        bitrate_kbps = min(default_bitrate_kbps, fitting_kbps)
        if bitrate_kbps < min_bitrate_kbps:
            # The default is already too big; can we still fit at the floor bitrate?
            floor_bytes = min_bitrate_kbps * 1000 * duration / 8
            if floor_bytes > max_bytes:
                raise AudioTooLargeError(
                    f"Audio is {duration / 60:.0f} min long; even at the minimum {min_bitrate_kbps} "
                    f"kbps it would exceed the {max_bytes / (1024 * 1024):.0f} MB upload limit. "
                    f"Split it into shorter parts."
                )
            bitrate_kbps = min_bitrate_kbps
    else:
        # Duration unknown: try the default bitrate and rely on the post-encode size check below.
        bitrate_kbps = default_bitrate_kbps

    fd, dst_path = tempfile.mkstemp(prefix="whispertyper_upload_", suffix=".mp3")
    os.close(fd)
    try:
        _encode_audio(exe, src_path, dst_path, max(1, round(bitrate_kbps)))
        out_size = os.path.getsize(dst_path)
        if out_size > max_bytes:
            # Duration was unknown, or a VBR/framing overshoot at the floor pushed us over.
            raise AudioTooLargeError(
                f"Compressed audio is {out_size / (1024 * 1024):.1f} MB, still above the "
                f"{max_bytes / (1024 * 1024):.0f} MB upload limit. The recording is too long to "
                f"transcribe in a single request; split it into shorter parts."
            )
    except Exception:
        try:
            os.remove(dst_path)
        except OSError:
            pass
        raise
    return dst_path, dst_path
