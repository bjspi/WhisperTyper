"""Native AAC tests with synthetic PCM only; never open a microphone or play audio."""
from __future__ import annotations

import math
import struct
from pathlib import Path

import pytest

from app.audio import native_encoder
from app.core import dsp
from app.core.env import is_MACOS, is_WINDOWS


def _movie_duration(data):
    """Read MP4 movie duration to catch encoder truncation/resampling mistakes."""
    offset = 0
    while offset + 8 <= len(data):
        size, kind = struct.unpack_from(">I4s", data, offset)
        assert size >= 8
        payload = data[offset + 8:offset + size]
        if kind == b"moov":
            return _movie_duration(payload)
        if kind == b"mvhd":
            assert payload[0] == 0
            timescale, duration = struct.unpack_from(">II", payload, 12)
            return duration / timescale
        offset += size
    raise AssertionError("Missing MP4 movie duration")


@pytest.mark.skipif(not (is_WINDOWS or is_MACOS), reason="Requires a native Windows/macOS encoder")
@pytest.mark.parametrize("bitrate", [48, 64, 96, 128, 160, 192])
def test_native_aac_preserves_duration_and_original(tmp_path, bitrate):
    if bitrate not in native_encoder.available_aac_bitrates():
        pytest.skip("Bitrate not offered by the installed native encoder")
    pcm = b"".join(struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * sample / 16000)))
                   for sample in range(3 * 16000))
    source = tmp_path / "source.wav"
    dsp.write_wav(str(source), pcm, 16000)
    original = source.read_bytes()
    destination = tmp_path / "upload.m4a"
    destination.touch()  # The upload worker reserves its temp filename before encoding.
    native_encoder.encode_wav_to_aac(str(source), str(destination), bitrate)
    encoded = destination.read_bytes()
    assert encoded[4:8] == b"ftyp"
    assert b"mdat" in encoded
    assert 3.0 <= _movie_duration(encoded) <= 3.3
    assert len(encoded) < len(original)
    assert source.read_bytes() == original


def test_unavailable_bitrate_rejected_before_writing(monkeypatch, tmp_path):
    monkeypatch.setattr(native_encoder, "available_aac_bitrates", lambda: (64,))
    destination = tmp_path / "upload.m4a"
    with pytest.raises(ValueError, match="bitrate unavailable"):
        native_encoder.encode_wav_to_aac("unused.wav", str(destination), 65)
    assert not destination.exists()


def test_macos_bitrates_are_probed_once_with_native_arguments(monkeypatch):
    calls = []

    def convert(arguments, **kwargs):
        calls.append(arguments)
        assert kwargs["check"]
        assert arguments[:7] == ["/usr/bin/afconvert", "-f", "m4af", "-d", "aac@48000", "-c", "1"]
        if arguments[8] == "48000":
            raise native_encoder.subprocess.CalledProcessError(1, arguments)
        Path(arguments[-1]).write_bytes(b"encoded")

    monkeypatch.setattr(native_encoder, "is_WINDOWS", False)
    monkeypatch.setattr(native_encoder, "is_MACOS", True)
    monkeypatch.setattr(native_encoder.os.path, "isfile", lambda _: True)
    monkeypatch.setattr(native_encoder.subprocess, "run", convert)
    native_encoder.available_aac_bitrates.cache_clear()
    try:
        assert native_encoder.available_aac_bitrates() == (64, 96, 128, 160, 192)
        assert native_encoder.available_aac_bitrates() == (64, 96, 128, 160, 192)
        assert len(calls) == 6
    finally:
        native_encoder.available_aac_bitrates.cache_clear()
