"""AAC encoder tests with synthetic PCM only; never open a microphone or play audio."""
from __future__ import annotations

import math
import struct

import pytest

from app.audio import aac_encoder
from app.core import dsp

av = pytest.importorskip("av")


def _sine_wav(path, samplerate=16000, seconds=3):
    pcm = b"".join(struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * sample / samplerate)))
                   for sample in range(seconds * samplerate))
    dsp.write_wav(str(path), pcm, samplerate)


@pytest.mark.parametrize("bitrate", aac_encoder.AAC_BITRATES)
def test_aac_preserves_duration_and_original(tmp_path, bitrate):
    source = tmp_path / "source.wav"
    _sine_wav(source)
    original = source.read_bytes()
    destination = tmp_path / "upload.m4a"
    destination.touch()  # The upload worker reserves its temp filename before encoding.
    aac_encoder.encode_wav_to_aac(str(source), str(destination), bitrate)
    encoded = destination.read_bytes()
    assert encoded[4:8] == b"ftyp"
    assert len(encoded) < len(original)
    assert source.read_bytes() == original
    with av.open(str(destination)) as container:
        stream = container.streams.audio[0]
        assert (stream.codec_context.name, stream.codec_context.profile) == ("aac", "LC")
        assert (stream.rate, stream.channels) == (aac_encoder.OUTPUT_SAMPLERATE, 1)
        assert 3.0 <= container.duration / av.time_base <= 3.1


def test_unavailable_bitrate_rejected_before_writing(tmp_path):
    destination = tmp_path / "upload.m4a"
    with pytest.raises(ValueError, match="bitrate unavailable"):
        aac_encoder.encode_wav_to_aac("unused.wav", str(destination), 65)
    assert not destination.exists()


def test_non_mono_pcm_rejected_and_source_kept(tmp_path):
    import wave

    source = tmp_path / "stereo.wav"
    with wave.open(str(source), "wb") as stereo:
        stereo.setnchannels(2)
        stereo.setsampwidth(2)
        stereo.setframerate(16000)
        stereo.writeframes(b"\x00\x00" * 3200)
    original = source.read_bytes()
    with pytest.raises(ValueError, match="mono 16-bit"):
        aac_encoder.encode_wav_to_aac(str(source), str(tmp_path / "upload.m4a"), 64)
    assert source.read_bytes() == original


def test_missing_encoder_offers_wav_only(monkeypatch):
    def broken_codec(*_args):
        raise av.codec.codec.UnknownCodecError("aac")

    monkeypatch.setattr(av.codec, "Codec", broken_codec)
    aac_encoder.available_aac_bitrates.cache_clear()
    try:
        assert aac_encoder.available_aac_bitrates() == ()
    finally:
        aac_encoder.available_aac_bitrates.cache_clear()
