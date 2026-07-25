"""Tests for the ffmpeg-independent decision logic in app/core/ffmpeg.py."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.core import ffmpeg


class TestIsVideoFile:
    @pytest.mark.parametrize("name", ["clip.mp4", "movie.MOV", "show.mkv", "x.webm", "y.ts"])
    def test_video_extensions(self, name: str):
        assert ffmpeg.is_video_file(name) is True

    @pytest.mark.parametrize("name", ["audio.mp3", "audio.wav", "audio.m4a", "noext", "x.mp3.txt"])
    def test_non_video_extensions(self, name: str):
        assert ffmpeg.is_video_file(name) is False


class TestAudioFileDecisions:
    @pytest.mark.parametrize("name", ["voice.ogg", "voice.OPUS", "memo.oga"])
    def test_ogg_and_opus_are_audio_that_need_normalization(self, name: str):
        assert ffmpeg.is_audio_file(name) is True
        assert ffmpeg.needs_audio_normalization(name) is True
        assert ffmpeg.needs_ffmpeg(name) is True

    @pytest.mark.parametrize("name", ["voice.mp3", "voice.wav", "voice.m4a", "voice.flac"])
    def test_native_audio_does_not_require_ffmpeg(self, name: str):
        assert ffmpeg.is_audio_file(name) is True
        assert ffmpeg.needs_audio_normalization(name) is False
        assert ffmpeg.needs_ffmpeg(name) is False

    def test_video_requires_ffmpeg(self):
        assert ffmpeg.needs_ffmpeg("meeting.mov") is True


class TestResolveFfmpeg:
    def test_configured_binary_path_wins(self, tmp_path: Path):
        exe = tmp_path / ("ffmpeg.exe" if ffmpeg.is_WINDOWS else "ffmpeg")
        exe.write_bytes(b"binary")
        exe.chmod(0o755)
        assert ffmpeg.resolve_ffmpeg(str(exe)) == str(exe)

    def test_configured_directory_is_expanded(self, tmp_path: Path):
        exe = tmp_path / ("ffmpeg.exe" if ffmpeg.is_WINDOWS else "ffmpeg")
        exe.write_bytes(b"binary")
        exe.chmod(0o755)
        assert ffmpeg.resolve_ffmpeg(str(tmp_path)) == str(exe)

    def test_quotes_are_stripped(self, tmp_path: Path):
        exe = tmp_path / ("ffmpeg.exe" if ffmpeg.is_WINDOWS else "ffmpeg")
        exe.write_bytes(b"binary")
        exe.chmod(0o755)
        assert ffmpeg.resolve_ffmpeg(f'"{exe}"') == str(exe)


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
