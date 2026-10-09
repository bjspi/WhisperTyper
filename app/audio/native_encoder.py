"""AAC/M4A through Windows Media Foundation or macOS Core Audio, without FFmpeg."""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import wave
from functools import lru_cache

from app.core import dsp
from app.core.env import is_MACOS, is_WINDOWS

_AFCONVERT = "/usr/bin/afconvert"
_AAC_BITRATES = (48, 64, 96, 128, 160, 192)


def _mac_aac(source: str, destination: str, bitrate_kbps: int) -> None:
    """Use Apple's included Core Audio converter with mono AAC at 48 kHz."""
    subprocess.run(
        [_AFCONVERT, "-f", "m4af", "-d", "aac@48000", "-c", "1",
         "-b", str(bitrate_kbps * 1000), source, destination],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


@lru_cache(maxsize=1)
def available_aac_bitrates() -> tuple[int, ...]:
    """Inspect the installed encoder once; unavailable systems offer WAV only."""
    try:
        if is_WINDOWS:
            from app.audio.windows_encoder import available_aac_bitrates as windows_bitrates
            return windows_bitrates()
        if is_MACOS and os.path.isfile(_AFCONVERT):
            # Probe synthetic silence during settings setup, never during capture/upload.
            supported = []
            with tempfile.TemporaryDirectory(prefix="whispertyper_codec_") as directory:
                source = os.path.join(directory, "probe.wav")
                dsp.write_wav(source, bytes(9600), 48000)
                for bitrate in _AAC_BITRATES:
                    destination = os.path.join(directory, f"probe_{bitrate}.m4a")
                    try:
                        _mac_aac(source, destination, bitrate)
                        if os.path.getsize(destination) > 0:
                            supported.append(bitrate)
                    except (OSError, subprocess.SubprocessError):
                        continue
            return tuple(supported)
    except OSError as error:
        logging.warning("Native AAC encoder unavailable: %s", error)
    return ()


def encode_wav_to_aac(source: str, destination: str, bitrate_kbps: int) -> None:
    """Encode a prepared mono PCM WAV; preserve the original on every failure."""
    if bitrate_kbps not in available_aac_bitrates():
        raise ValueError("AAC bitrate unavailable in the installed native encoder; choose WAV or another bitrate.")
    with wave.open(source, "rb") as recording:
        if recording.getnchannels() != 1 or recording.getsampwidth() != 2:
            raise ValueError("Native AAC encoding requires mono 16-bit PCM.")
        samplerate = recording.getframerate()
        pcm = recording.readframes(recording.getnframes()) if is_WINDOWS else b""
    if is_WINDOWS:
        from app.audio.windows_encoder import write_aac
        write_aac(destination, pcm, samplerate, bitrate_kbps)
    elif is_MACOS:
        _mac_aac(source, destination, bitrate_kbps)
    else:
        raise OSError("Native AAC encoding is supported only on Windows and macOS.")
