"""FFmpeg detection, video→audio extraction, upload-size compression and splitting.

Single responsibility: locate a usable ``ffmpeg`` binary and turn whatever file the user picked
into an MP3 the transcription API will accept — extracting audio from video containers and, when
a file is above the endpoint's upload limit, shrinking it enough to fit. No Qt, no app state —
callers pass the configured path plus limits and get back plain values / a temp file path.

Whisper-style endpoints reject video containers (or silently misbehave on them), and even
audio-capable containers can be too large to upload. We re-encode to mono, 16 kHz MP3 — Whisper
resamples to 16 kHz mono internally anyway, so this costs no transcription quality while shrinking
the file dramatically. The bitrate stays at a comfortable default and is only lowered (down to a
configurable floor) when the file would otherwise exceed the upload limit. A file too long for one
upload even at that floor is split into several uploads, cut at pauses in the speech.
"""
from __future__ import annotations

import logging
import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

from app.core.env import is_WINDOWS
from app.platform.system import no_window_kwargs

# Encode target: mono, 16 kHz MP3. Whisper works at 16 kHz mono internally, so this is lossless for
# transcription while keeping files small. The default bitrate is used unless the file must shrink
# further to fit under the upload limit.
_TARGET_SAMPLE_RATE = 16000
_DEFAULT_BITRATE_KBPS = 128
#: The bitrates MP3 offers at 16 kHz (MPEG-2 Layer III). LAME rounds any other value to the nearest
#: of these, which can be above the one that fits, so a computed bitrate is rounded down to one.
_MP3_BITRATES_KBPS = (8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160)
# Aim a little below the hard limit so container/MP3 framing overhead can't push us over it.
_SIZE_SAFETY = 0.95
#: Longest an encode or the pause scan of a long file may take.
_FFMPEG_TIMEOUT_S = 600
#: Parts are planned at this share of the longest part that fits, so moving a cut to a nearby
#: pause cannot push a part over the limit.
_PART_FILL = 0.9
#: A cut moves to the longest pause within this distance of its even position (at most this
#: share of the longest part, for small upload limits).
_PAUSE_SEARCH_S = 60.0
_PAUSE_SEARCH_SHARE = 0.05
#: What counts as a pause in the speech: quieter than this for at least this long.
_PAUSE_NOISE_DB = -35
_PAUSE_MIN_S = 0.4
_SILENCE_START = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
_SILENCE_END = re.compile(r"silence_end:\s*(\d+(?:\.\d+)?)")


class AudioTooLargeError(RuntimeError):
    """The file can't be brought under the upload limit (too long even at the minimum bitrate)."""


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
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10, **no_window_kwargs(),
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
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, **no_window_kwargs(),
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

def _encode_audio(exe: str, src_path: str, dst_path: str, bitrate_kbps: int,
                  start_s: Optional[float] = None, length_s: Optional[float] = None) -> None:
    """Re-encode ``src_path`` to a mono, 16 kHz MP3 at ``bitrate_kbps`` (dropping any video track).

    ``start_s``/``length_s`` encode only that stretch of the source (one part of a split file).

    Raises:
        RuntimeError: If ffmpeg exits non-zero or produces no output.
    """
    excerpt: List[str] = []
    if start_s is not None and length_s is not None:
        excerpt = ["-ss", f"{start_s:.3f}", "-t", f"{length_s:.3f}"]
    cmd: List[str] = [
        exe, "-y",                          # overwrite the temp file we just created
        *excerpt,                           # before -i: seek in the input instead of decoding up to it
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
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=_FFMPEG_TIMEOUT_S,
            **no_window_kwargs(),
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise RuntimeError(f"ffmpeg could not be run: {e}")

    if result.returncode != 0:
        # ffmpeg writes diagnostics to stderr; surface the tail so failures are debuggable.
        tail = (result.stderr or "").strip().splitlines()[-3:]
        raise RuntimeError("ffmpeg failed to encode audio:\n" + "\n".join(tail))
    if not os.path.isfile(dst_path) or os.path.getsize(dst_path) == 0:
        raise RuntimeError("ffmpeg produced no audio output (the file may have no audio track).")


def _encode_to_temp(exe: str, src_path: str, bitrate_kbps: float, max_bytes: int,
                    start_s: Optional[float] = None, length_s: Optional[float] = None) -> str:
    """Encode (a stretch of) ``src_path`` into a temp MP3 the caller must delete; it must fit ``max_bytes``.

    Raises:
        AudioTooLargeError: The encoded file is still above ``max_bytes``.
        RuntimeError: If ffmpeg fails while encoding.
    """
    fd, dst_path = tempfile.mkstemp(prefix="whispertyper_upload_", suffix=".mp3")
    os.close(fd)
    try:
        _encode_audio(exe, src_path, dst_path, _mp3_bitrate(bitrate_kbps), start_s, length_s)
        out_size = os.path.getsize(dst_path)
        if out_size > max_bytes:
            # Duration was unknown, or a VBR/framing overshoot at the floor pushed us over.
            raise AudioTooLargeError(
                f"Compressed audio is {out_size / (1024 * 1024):.1f} MB, still above the "
                f"{max_bytes / (1024 * 1024):.0f} MB upload limit. The recording is too long to "
                f"transcribe in a single request; split it into shorter parts."
            )
    except Exception:
        _remove_quietly(dst_path)
        raise
    return dst_path


def _mp3_bitrate(bitrate_kbps: float) -> int:
    """The highest MP3 bitrate at 16 kHz that does not exceed ``bitrate_kbps`` (the lowest one at least)."""
    fitting = [rate for rate in _MP3_BITRATES_KBPS if rate <= bitrate_kbps]
    return fitting[-1] if fitting else _MP3_BITRATES_KBPS[0]


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _size(path: str) -> Optional[int]:
    """Size in bytes, or None when the file cannot be read."""
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def _fitting_kbps(max_bytes: int, duration_s: float) -> float:
    """The bitrate whose encode of ``duration_s`` lands right at the safety-adjusted limit."""
    return (max_bytes * _SIZE_SAFETY * 8) / duration_s / 1000


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
    below ``min_bitrate_kbps`` — when the default wouldn't fit under ``max_bytes``. A file that
    needs more than one upload (see ``upload_parts``) goes through ``split_upload`` instead.

    Returns:
        (upload_path, temp_path). ``temp_path`` is None when the original file is uploaded as-is;
        otherwise it equals ``upload_path`` and the caller must remove it once done.

    Raises:
        AudioTooLargeError: The file can't be shrunk under ``max_bytes`` (too long even at the floor),
            or it's oversized and no ffmpeg is available to compress it.
        RuntimeError: If ffmpeg fails while encoding.
    """
    src_size = _size(src_path)
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
        bitrate_kbps = min(default_bitrate_kbps, _fitting_kbps(max_bytes, duration))
        if bitrate_kbps < min_bitrate_kbps:
            # The default is already too big; can we still fit at the floor bitrate?
            if upload_parts_for(duration, max_bytes, min_bitrate_kbps) > 1:
                raise AudioTooLargeError(
                    f"Audio is {duration / 60:.0f} min long; even at the minimum {min_bitrate_kbps} "
                    f"kbps it would exceed the {max_bytes / (1024 * 1024):.0f} MB upload limit. "
                    f"Split it into shorter parts."
                )
            bitrate_kbps = min_bitrate_kbps
    else:
        # Duration unknown: try the default bitrate and rely on the post-encode size check.
        bitrate_kbps = default_bitrate_kbps

    dst_path = _encode_to_temp(exe, src_path, bitrate_kbps, max_bytes)
    return dst_path, dst_path


# --- Splitting a file that is too long for one upload ---------------------------------------
def max_part_seconds(max_bytes: int, min_bitrate_kbps: int) -> float:
    """The longest audio that still fits under the safety-adjusted ``max_bytes`` at the minimum bitrate."""
    return (max_bytes * _SIZE_SAFETY * 8) / (min_bitrate_kbps * 1000)


def upload_parts_for(duration_s: float, max_bytes: int, min_bitrate_kbps: int) -> int:
    """How many uploads ``duration_s`` of audio takes: 1 while it fits at the minimum bitrate."""
    if min_bitrate_kbps * 1000 * duration_s / 8 <= max_bytes:
        return 1
    return math.ceil(duration_s / (max_part_seconds(max_bytes, min_bitrate_kbps) * _PART_FILL))


def upload_parts(exe: Optional[str], src_path: str, *, transcode_source: bool, max_bytes: int,
                 min_bitrate_kbps: int) -> int:
    """How many uploads the picked file takes (see ``upload_parts_for``).

    Only files that are re-encoded anyway (``transcode_source``) or exceed ``max_bytes`` are
    measured. Without ffmpeg, or when it can't tell the duration, the answer is 1 and
    ``prepare_upload`` reports whatever goes wrong.
    """
    size = _size(src_path)
    if not exe or (not transcode_source and (size is None or size <= max_bytes)):
        return 1
    duration = probe_duration(exe, src_path)
    if not duration:
        return 1
    return upload_parts_for(duration, max_bytes, min_bitrate_kbps)


@dataclass(frozen=True)
class Pause:
    """A stretch without speech, in seconds from the start of the file."""

    start: float
    end: float

    @property
    def middle(self) -> float:
        """The cut position inside the pause."""
        return (self.start + self.end) / 2

    @property
    def length(self) -> float:
        """Seconds without speech; a longer pause more likely ends a sentence."""
        return self.end - self.start


def detect_pauses(exe: str, src_path: str) -> List[Pause]:
    """The pauses in ``src_path``'s speech (ffmpeg's ``silencedetect``); none when the scan fails."""
    cmd = [exe, "-hide_banner", "-nostats", "-i", src_path, "-vn", "-ac", "1",
           "-af", f"silencedetect=noise={_PAUSE_NOISE_DB}dB:d={_PAUSE_MIN_S}", "-f", "null", "-"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=_FFMPEG_TIMEOUT_S,
            **no_window_kwargs(),
        )
    except (OSError, subprocess.SubprocessError) as e:
        logging.warning("ffmpeg pause scan failed for %s, cutting at even positions: %s",
                        os.path.basename(src_path), e)
        return []
    return parse_pauses(result.stderr or "")


def parse_pauses(silencedetect_output: str) -> List[Pause]:
    """The pauses reported by ``silencedetect`` (a ``silence_start`` line, then its ``silence_end``)."""
    pauses: List[Pause] = []
    start: Optional[float] = None
    for line in silencedetect_output.splitlines():
        started = _SILENCE_START.search(line)
        if started:
            start = max(0.0, float(started.group(1)))
            continue
        ended = _SILENCE_END.search(line)
        if ended and start is not None:
            pauses.append(Pause(start, float(ended.group(1))))
            start = None
    return pauses


def plan_cuts(duration_s: float, parts: int, pauses: List[Pause], max_part_s: float) -> List[float]:
    """Where to cut ``duration_s`` into ``parts`` similar lengths, each cut in the longest nearby pause.

    Every cut aims at an even share of what is left, so a cut moved to a pause does not shift the
    following ones. Without a pause nearby the cut stays at the even position (a word may be split).
    """
    window = min(_PAUSE_SEARCH_S, max_part_s * _PAUSE_SEARCH_SHARE)
    cuts: List[float] = []
    start = 0.0
    for remaining in range(parts, 1, -1):
        target = start + (duration_s - start) / remaining
        nearby = [pause for pause in pauses
                  if abs(pause.middle - target) <= window and start < pause.middle <= start + max_part_s]
        cut = max(nearby, key=lambda pause: pause.length).middle if nearby else target
        cuts.append(cut)
        start = cut
    return cuts


def split_upload(
    exe: str,
    src_path: str,
    *,
    parts: int,
    max_bytes: int,
    min_bitrate_kbps: int,
    default_bitrate_kbps: int = _DEFAULT_BITRATE_KBPS,
) -> List[str]:
    """Encode ``src_path`` as ``parts`` consecutive MP3 uploads cut at pauses; the caller deletes them.

    Each part keeps the default bitrate when it fits and is only lowered towards
    ``min_bitrate_kbps`` as far as its length needs.

    Raises:
        AudioTooLargeError: A part is still above ``max_bytes``.
        RuntimeError: If ffmpeg cannot tell the duration or fails while encoding.
    """
    duration = probe_duration(exe, src_path)
    if not duration:
        raise RuntimeError("ffmpeg could not read the length of the audio, so it cannot be split into parts.")
    cuts = plan_cuts(duration, parts, detect_pauses(exe, src_path), max_part_seconds(max_bytes, min_bitrate_kbps))
    bounds = list(zip([0.0, *cuts], [*cuts, duration]))
    logging.info("Splitting %s (%.0f s) into %d parts at %s s", os.path.basename(src_path), duration,
                 len(bounds), ", ".join(f"{cut:.1f}" for cut in cuts))
    paths: List[str] = []
    try:
        for start, end in bounds:
            length = end - start
            bitrate = max(min_bitrate_kbps, min(default_bitrate_kbps, _fitting_kbps(max_bytes, length)))
            paths.append(_encode_to_temp(exe, src_path, bitrate, max_bytes, start, length))
    except Exception:
        for path in paths:
            _remove_quietly(path)
        raise
    return paths
