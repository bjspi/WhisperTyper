"""Tests for the PCM helpers in app/core/dsp.py."""
from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

from app.core import dsp


def make_tone(samplerate: int = 16000, duration_s: float = 0.1, amplitude: int = 1000,
              frequency: float = 440.0) -> bytes:
    frames = bytearray()
    for i in range(int(samplerate * duration_s)):
        sample = int(amplitude * math.sin(2 * math.pi * frequency * i / samplerate))
        frames += struct.pack("<h", sample)
    return bytes(frames)


class TestPeak:
    def test_silence(self):
        assert dsp.peak(b"\x00" * 320) == 0

    def test_known_amplitude(self):
        pcm = struct.pack("<3h", 100, -2000, 500)
        assert dsp.peak(pcm) == 2000


class TestDurationSeconds:
    def test_one_second_of_16k_mono(self):
        pcm = b"\x00" * (16000 * 2)
        assert dsp.duration_seconds(pcm, 16000) == 1.0

    def test_zero_rate_is_safe(self):
        assert dsp.duration_seconds(b"\x00\x00", 0) == 0.0


class TestApplyGain:
    def test_positive_gain_raises_peak(self):
        pcm = make_tone(amplitude=1000)
        louder = dsp.apply_gain(pcm, 6.0)  # +6 dB ≈ ×2
        assert dsp.peak(louder) > dsp.peak(pcm) * 1.8

    def test_gain_clips_instead_of_overflowing(self):
        pcm = make_tone(amplitude=30000)
        boosted = dsp.apply_gain(pcm, 20.0)
        # int16 range is [-32768, 32767]; peak() reports the absolute value, so a
        # negative full-scale clip legitimately shows up as 32768.
        assert dsp.peak(boosted) <= 32768


class TestResample:
    def test_downsampling_halves_length(self):
        pcm = make_tone(samplerate=44100, duration_s=0.5)
        out = dsp.resample(pcm, 44100, 16000)
        expected = len(pcm) * 16000 / 44100
        assert abs(len(out) - expected) < 64  # small converter slack


class TestWriteWav:
    def test_roundtrip(self, tmp_path: Path):
        pcm = make_tone()
        path = str(tmp_path / "out.wav")
        dsp.write_wav(path, pcm, 16000)
        with wave.open(path, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.readframes(wf.getnframes()) == pcm


class TestPrepareForUpload:
    def test_resamples_to_target_rate_and_applies_gain(self):
        pcm = make_tone(samplerate=44100, duration_s=0.5, amplitude=1000)
        out, rate = dsp.prepare_for_upload(pcm, 44100, 16000, 6.0)
        assert rate == 16000
        assert abs(dsp.duration_seconds(out, rate) - 0.5) < 0.01
        assert dsp.peak(out) > 1800

    def test_keeps_source_rate_when_resampling_fails(self, monkeypatch):
        monkeypatch.setattr(dsp, "resample", lambda *_args: (_ for _ in ()).throw(ValueError("unsupported")))
        pcm = make_tone(samplerate=44100)
        assert dsp.prepare_for_upload(pcm, 44100, 16000, 0.0) == (pcm, 44100)


class TestNormalizingGain:
    def test_lifts_quiet_recording_to_target_peak(self):
        pcm = struct.pack("<2h", 3277, -100)  # peak -20 dBFS
        assert math.isclose(dsp.normalizing_gain_db(pcm, 30.0), 19.0, abs_tol=0.05)

    def test_capped_at_limit(self):
        pcm = struct.pack("<2h", 33, -10)  # peak -60 dBFS
        assert dsp.normalizing_gain_db(pcm, 15.0) == 15.0

    def test_loud_recording_is_never_attenuated(self):
        assert dsp.normalizing_gain_db(struct.pack("<h", 32767), 15.0) == 0.0

    def test_silence_gets_no_gain(self):
        assert dsp.normalizing_gain_db(b"\x00" * 320, 15.0) == 0.0

    def test_normalized_upload_does_not_clip(self):
        pcm = make_tone(amplitude=10000)
        out, _rate = dsp.prepare_for_upload(pcm, 16000, 16000, 20.0, normalize=True)
        assert 0.85 * 32767 < dsp.peak(out) < 32767


class TestBlockDbfs:
    def test_full_scale_square_is_about_zero(self):
        pcm = struct.pack("<4h", 32767, -32767, 32767, -32767)
        assert dsp.block_dbfs(pcm)[0] > -0.1

    def test_silence_is_floor_and_parts_split(self):
        assert dsp.block_dbfs(b"\x00" * 400, 2) == [-96.0, -96.0]
