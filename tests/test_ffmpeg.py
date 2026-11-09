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
