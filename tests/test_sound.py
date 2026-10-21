"""Start/stop sounds: off opens no output stream; on Windows a keepalive keeps the output awake."""
from __future__ import annotations

import threading
import time
import wave

import pytest

from app.audio import sound as sound_module
from app.audio.sound import SoundPlayer


class FakeStream:
    def __init__(self):
        """Record every write."""
        self.writes: list[bytes] = []
        self.closed = False
        self.wrote = threading.Event()

    def write(self, data):
        self.writes.append(data)
        self.wrote.set()

    def stop_stream(self):
        pass

    def close(self):
        self.closed = True


class FakePyAudio:
    opened: list[FakeStream] = []

    def get_format_from_width(self, width):
        return width

    def open(self, **_kwargs):
        stream = FakeStream()
        FakePyAudio.opened.append(stream)
        return stream

    def terminate(self):
        pass


@pytest.fixture
def player(tmp_path, monkeypatch):
    FakePyAudio.opened = []
    monkeypatch.setattr(sound_module.pyaudio, "PyAudio", FakePyAudio)
    (tmp_path / "resources").mkdir()
    with wave.open(str(tmp_path / "resources" / "beep.wav"), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x01\x00" * 160)
    player = SoundPlayer(lambda *parts: str(tmp_path.joinpath(*parts)))
    yield player
    player.close()


def test_turned_off_opens_no_stream_and_plays_nothing(player):
    player.set_enabled(False)
    player.preload(["beep.wav"])
    player.play("beep.wav")
    time.sleep(0.05)
    assert FakePyAudio.opened == []

    player.set_enabled(True)
    assert len(FakePyAudio.opened) == 1
    player.set_enabled(False)
    assert FakePyAudio.opened[0].closed


def test_windows_keepalive_writes_short_silence(player, monkeypatch):
    monkeypatch.setattr(sound_module, "is_WINDOWS", True)
    monkeypatch.setattr(sound_module, "_KEEPALIVE_INTERVAL_S", 0.01)
    player.preload(["beep.wav"])
    stream = FakePyAudio.opened[0]
    assert stream.wrote.wait(2)
    assert stream.writes[0] == b"\x00" * int(16000 * sound_module._KEEPALIVE_SILENCE_S) * 2
    player.close()
    count = len(stream.writes)
    time.sleep(0.05)
    assert len(stream.writes) == count  # Stopped with close().
