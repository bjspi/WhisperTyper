"""Tests for app/audio/playback.py (WavPlayer) with a fake PyAudio."""
from __future__ import annotations

import threading
import time
import wave
from pathlib import Path
from typing import Any, List

import pytest

pytest.importorskip("pyaudio")  # Imported by app.audio.playback; the playback itself uses a fake below.

from app.audio.playback import WavPlayer  # noqa: E402


class FakeStream:
    def __init__(self, written: List[bytes], delay: float) -> None:
        """Record writes into ``written``, sleeping ``delay`` per chunk."""
        self.written, self.delay, self.closed = written, delay, False

    def write(self, frames: bytes) -> None:
        time.sleep(self.delay)
        self.written.append(frames)

    def stop_stream(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class FakeAudio:
    def __init__(self, delay: float = 0.0) -> None:
        """Streams share one write log; ``delay`` slows each write."""
        self.written: List[bytes] = []
        self.delay = delay
        self.streams: List[FakeStream] = []

    def get_format_from_width(self, width: int) -> int:
        return width

    def open(self, **_kwargs: Any) -> FakeStream:
        stream = FakeStream(self.written, self.delay)
        self.streams.append(stream)
        return stream


def make_wav(path: Path, frames: int) -> str:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * frames)
    return str(path)


def test_plays_whole_file_and_reports_the_end(tmp_path: Path):
    audio = FakeAudio()
    player = WavPlayer(lambda: audio)
    done = threading.Event()
    player.play(make_wav(tmp_path / "a.wav", 5000), done.set)
    assert done.wait(2)
    assert sum(len(chunk) for chunk in audio.written) == 10000
    assert audio.streams[0].closed


def test_stop_ends_playback_early(tmp_path: Path):
    audio = FakeAudio(delay=0.02)
    player = WavPlayer(lambda: audio)
    done = threading.Event()
    player.play(make_wav(tmp_path / "long.wav", 16000 * 10), done.set)
    time.sleep(0.05)
    assert player.is_playing()
    player.stop()
    assert done.wait(2)
    assert not player.is_playing()
    assert sum(len(chunk) for chunk in audio.written) < 16000 * 2 * 10
