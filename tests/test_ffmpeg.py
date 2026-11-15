"""Tests for the ffmpeg-independent decision logic in app/services/ffmpeg.py."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.services import ffmpeg


class TestResolveFfmpeg:
    @pytest.mark.parametrize("setting", ["{exe}", "{folder}", '"{exe}"'], ids=["binary", "folder", "quoted"])
    def test_configured_path_is_found(self, tmp_path: Path, setting: str):
        exe = tmp_path / ("ffmpeg.exe" if ffmpeg.is_WINDOWS else "ffmpeg")
        exe.write_bytes(b"binary")
        exe.chmod(0o755)
        assert ffmpeg.resolve_ffmpeg(setting.format(exe=exe, folder=tmp_path)) == str(exe)


class TestPrepareUpload:
    def test_small_audio_passes_through_untouched(self, tmp_path: Path):
        src = tmp_path / "small.mp3"
        src.write_bytes(b"x" * 1024)
        upload_path, temp = ffmpeg.prepare_upload(
            None, str(src), transcode_source=False,
            max_bytes=24 * 1024 * 1024, min_bitrate_kbps=80,
        )
        assert upload_path == str(src)
        assert temp is None

    def test_oversized_audio_without_ffmpeg_is_rejected(self, tmp_path: Path):
        src = tmp_path / "big.mp3"
        src.write_bytes(b"x" * 2048)
        with pytest.raises(ffmpeg.AudioTooLargeError):
            ffmpeg.prepare_upload(
                None, str(src), transcode_source=False,
                max_bytes=1024, min_bitrate_kbps=80,
            )


def test_duration_probe_reads_ffmpeg_output_whatever_the_file_name(tmp_path: Path):
    """Regression: ffmpeg prints the path in UTF-8; decoding it as cp1252 lost the duration ("ā" is 0xC4 0x81)."""
    import wave

    exe = ffmpeg.resolve_ffmpeg()
    if not exe:
        pytest.skip("ffmpeg is not installed")
    path = tmp_path / "Interview mit Rā.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000)
    assert ffmpeg.probe_duration(exe, str(path)) == pytest.approx(1.0, abs=0.05)


class TestSplitPlanning:
    def test_a_file_that_fits_at_the_minimum_bitrate_is_one_upload(self):
        # 24 MB at 80 kbps holds about 41 minutes.
        assert ffmpeg.upload_parts_for(40 * 60, 24 * 1024 * 1024, 80) == 1

    def test_a_longer_file_needs_parts_that_each_fit(self):
        max_bytes = 24 * 1024 * 1024
        parts = ffmpeg.upload_parts_for(3 * 3600, max_bytes, 80)
        assert parts > 1
        assert 3 * 3600 / parts <= ffmpeg.max_part_seconds(max_bytes, 80)

    def test_parses_the_pauses_silencedetect_reports(self):
        output = ("[silencedetect @ 0x1] silence_start: -0.01\n"
                  "[silencedetect @ 0x1] silence_end: 1.5 | silence_duration: 1.51\n"
                  "size=N/A time=00:00:10\n"
                  "[silencedetect @ 0x1] silence_start: 7.25\n"
                  "[silencedetect @ 0x1] silence_end: 8.0 | silence_duration: 0.75\n"
                  "[silencedetect @ 0x1] silence_start: 9.5\n")  # still silent at the end: no pause
        assert ffmpeg.parse_pauses(output) == [ffmpeg.Pause(0.0, 1.5), ffmpeg.Pause(7.25, 8.0)]

    def test_cuts_move_to_the_longest_nearby_pause(self):
        pauses = [ffmpeg.Pause(990.0, 991.0), ffmpeg.Pause(1010.0, 1013.0), ffmpeg.Pause(1500.0, 1510.0)]
        cuts = ffmpeg.plan_cuts(3000.0, 3, pauses, max_part_s=2391.0)
        assert cuts[0] == pytest.approx(1011.5)  # the longer of the two pauses near 1000 s, not the far one
        assert cuts[1] == pytest.approx(1011.5 + (3000.0 - 1011.5) / 2)  # no pause nearby: even position

    @pytest.mark.parametrize(("computed", "sent"), [(92.5, 80), (128, 128), (125.4, 112), (300, 160), (5, 8)])
    def test_bitrates_round_down_to_one_mp3_offers(self, computed: float, sent: int):
        # LAME would round 92.5 up to 96 kbps, above the bitrate that fits the upload limit.
        assert ffmpeg._mp3_bitrate(computed) == sent

    def test_without_pauses_the_cuts_are_even(self):
        assert ffmpeg.plan_cuts(300.0, 3, [], max_part_s=200.0) == pytest.approx([100.0, 200.0])


def test_split_upload_cuts_a_file_into_parts_that_fit(tmp_path: Path):
    import math
    import wave

    exe = ffmpeg.resolve_ffmpeg()
    if not exe:
        pytest.skip("ffmpeg is not installed")
    path = tmp_path / "long.wav"
    rate = 16000
    tone = b"".join(int(8000 * math.sin(2 * math.pi * 440 * i / rate)).to_bytes(2, "little", signed=True)
                    for i in range(rate * 2))
    silence = b"\x00\x00" * rate
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(tone + silence + tone + silence + tone)  # 8 s with pauses at 2-3 s and 5-6 s
    max_bytes = 40 * 1024  # 4 s at 80 kbps
    parts = ffmpeg.upload_parts(exe, str(path), transcode_source=False, max_bytes=max_bytes, min_bitrate_kbps=80)
    assert parts == 3
    uploads = ffmpeg.split_upload(exe, str(path), parts=parts, max_bytes=max_bytes, min_bitrate_kbps=80)
    try:
        assert len(uploads) == 3
        assert all(0 < Path(upload).stat().st_size <= max_bytes for upload in uploads)
    finally:
        for upload in uploads:
            Path(upload).unlink()
