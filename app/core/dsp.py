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

#: Peak level a normalised recording is lifted to; 1 dB of headroom keeps lossy encoding from clipping.
NORMALIZE_TARGET_DBFS = -1.0


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


def meter_level(pcm: bytes) -> float:
    """Tray-meter level 0..1 from a 16-bit PCM block; boosted so normal speech fills the meter."""
    return min(1.0, peak(pcm) / 32767.0 * 1.8)


def block_dbfs(pcm: bytes, parts: int = 1) -> list[float]:
    """RMS level in dBFS of ``parts`` equal slices of a 16-bit block; digital silence is -96 dBFS.

    An absolute scale (no normalisation to the loudest block), so background noise stays low.
    """
    samples = len(pcm) // 2
    step = max(1, samples // max(1, parts)) * 2
    levels = []
    for start in range(0, samples * 2 - step + 1, step):
        rms = audioop.rms(pcm[start:start + step], 2)
        levels.append(20.0 * math.log10(rms / 32768.0) if rms > 0 else -96.0)
    return levels


def normalizing_gain_db(pcm: bytes, max_gain_db: float, target_dbfs: float = NORMALIZE_TARGET_DBFS) -> float:
    """Gain that lifts the recording's peak to ``target_dbfs``, at most ``max_gain_db``, never below 0.

    One constant gain for the whole recording, measured after it ended: quiet passages (such as
    silence before the first word) do not change it, and the loudest moment never clips.
    """
    peak_value = peak(pcm)
    if peak_value <= 0 or max_gain_db <= 0:
        return 0.0
    headroom = target_dbfs - 20.0 * math.log10(peak_value / 32767.0)
    return max(0.0, min(max_gain_db, headroom))


def prepare_for_upload(pcm: bytes, source_rate: int, target_rate: int, gain_db: float,
                       normalize: bool = False) -> tuple[bytes, int]:
    """Resample to ``target_rate`` (keeping the source rate if that fails) and apply gain.

    With ``normalize``, ``gain_db`` is the upper limit of a peak-normalising gain.
    """
    rate = int(source_rate or target_rate)
    if rate != target_rate:
        try:
            pcm = resample(pcm, rate, target_rate)
            logging.info(f"Resampled recording from {rate} Hz to {target_rate} Hz before upload.")
            rate = target_rate
        except Exception as e:
            logging.warning(f"Could not resample recording from {rate} Hz to {target_rate} Hz: {e}")
    if normalize:
        limit = gain_db
        gain_db = normalizing_gain_db(pcm, limit)
        logging.info(f"Normalized recording: +{gain_db:.1f} dB (limit +{limit} dB, "
                     f"peak target {NORMALIZE_TARGET_DBFS} dBFS).")
        return apply_gain(pcm, gain_db), rate
    if gain_db > 0:
        logging.info(f"Applied +{gain_db} dB gain to recording.")
    return apply_gain(pcm, gain_db), rate
