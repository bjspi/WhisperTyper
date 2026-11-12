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
