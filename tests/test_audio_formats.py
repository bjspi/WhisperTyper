"""Upload decisions for picked media files (no FFmpeg or network needed)."""
from __future__ import annotations

import pytest

from app.core import audio_formats


@pytest.mark.parametrize("name, expected", [
    ("voice.opus", ("voice.ogg", "audio/ogg")),
    ("voice.oga", ("voice.ogg", "audio/ogg")),
    ("talk.ogg", ("talk.ogg", "audio/ogg")),
    ("talk.FLAC", ("talk.FLAC", "audio/flac")),
    ("talk.mpga", ("talk.mpga", "audio/mpeg")),
    ("clip.webm", ("clip.webm", "audio/webm")),
])
def test_upload_name_and_type_do_not_depend_on_the_os_mime_database(name, expected):
    assert (audio_formats.upload_filename(name), audio_formats.upload_content_type(name)) == expected


def test_only_videos_the_apis_reject_require_ffmpeg():
    assert audio_formats.requires_ffmpeg("movie.mkv")
    assert audio_formats.requires_ffmpeg("movie.mov")
    assert not audio_formats.requires_ffmpeg("clip.mp4")  # accepted as uploaded
    assert not audio_formats.requires_ffmpeg("voice.opus")  # sent as .ogg without FFmpeg


def test_rejected_audio_extensions_are_converted_when_ffmpeg_exists():
    assert audio_formats.needs_audio_normalization("voice.opus")
    assert audio_formats.needs_audio_normalization("song.aac")
    assert not audio_formats.needs_audio_normalization("talk.ogg")
