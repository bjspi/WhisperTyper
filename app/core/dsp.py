"""PCM DSP helpers for 16-bit mono audio — C-accelerated via ``audioop``.

Single responsibility: transform raw PCM buffers (gain / resample / peak) and write WAV.
No Qt, no device I/O, no app state — pure functions, unit-testable.

Note: ``apply_gain`` uses ``audioop.mul``, which multiplies in C and silently clips
on overflow.
"""

from __future__ import annotations

import audioop
import logging
import math
import struct
import wave


def peak(pcm: bytes) -> int:
    """Absolute peak sample value of a 16-bit PCM buffer (0 for empty/silence)."""
    if not pcm:
        return 0
    try:
        return audioop.max(pcm, 2)
    except Exception:
        # Extremely defensive fallback; audioop.max is virtually always available.
        m = 0
        for i in range(0, len(pcm) - 1, 2):
            a = abs(struct.unpack_from("<h", pcm, i)[0])
            if a > m:
                m = a
        return m


def duration_seconds(pcm: bytes, samplerate: int, sampwidth: int = 2, channels: int = 1) -> float:
    """Playback length in seconds of a raw PCM buffer (0.0 for empty/invalid rate)."""
    frame_bytes = max(1, sampwidth * channels)
    rate = int(samplerate) if samplerate else 0
    if rate <= 0:
        return 0.0
    return len(pcm) / frame_bytes / rate


def apply_gain(pcm: bytes, gain_db: float) -> bytes:
    """Amplify 16-bit PCM by ``gain_db`` decibels (clipped). No-op for gain <= 0."""
    if gain_db <= 0 or not pcm:
        return pcm
    factor = math.pow(10.0, gain_db / 20.0)
    try:
        return audioop.mul(pcm, 2, factor)
    except Exception as e:
        logging.error(f"Gain application failed: {e}")
        return pcm


def resample(pcm: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resample 16-bit mono PCM from ``from_rate`` to ``to_rate`` Hz."""
    if not pcm or from_rate == to_rate:
        return pcm
    converted, _ = audioop.ratecv(pcm, 2, 1, from_rate, to_rate, None)
    return converted


def write_wav(path: str, pcm: bytes, samplerate: int, channels: int = 1, sampwidth: int = 2) -> None:
    """Write raw PCM to a WAV file."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(samplerate)
        wf.writeframes(pcm)
