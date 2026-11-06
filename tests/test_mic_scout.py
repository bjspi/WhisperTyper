"""The first-run microphone search: it only switches when the default is silent and another device hears."""
from __future__ import annotations

import struct
from typing import Any, Dict, List
from unittest.mock import Mock

import pytest

pytest.importorskip("pyaudio")

from app.audio import mic_scout  # noqa: E402
from app.audio.mic_scout import MicrophoneScout, ScoutDevice, scout_devices  # noqa: E402

CHUNK = 8
LOUD = 8000  # about -12 dBFS, a voice
FAINT = 40  # about -58 dBFS, background noise
HEADSET = {"index": 2, "name": "Headset", "label": "Headset Microphone"}


def pcm(peak: int) -> bytes:
    return struct.pack(f"<{CHUNK}h", *([peak] * CHUNK))


class FakeStream:
    """Delivers the given peaks, then nothing more."""

    def __init__(self, peaks: List[int]) -> None:
        """Queue the block peaks to deliver."""
        self._peaks = list(peaks)
        self.closed = False

    def get_read_available(self) -> int:
        return CHUNK if self._peaks else 0

    def read(self, _frames: int, exception_on_overflow: bool = False) -> bytes:
        assert exception_on_overflow is False
        return pcm(self._peaks.pop(0))

    def close(self) -> None:
        self.closed = True


def scout_with(monkeypatch: pytest.MonkeyPatch, devices: List[Dict[str, Any]], streams: Dict[int, FakeStream]) -> tuple:
    """A scout over fake devices whose search ends quickly, with its two callbacks."""
    monkeypatch.setattr(mic_scout, "SILENCE_GRACE_S", 0.0)
    monkeypatch.setattr(mic_scout, "LISTEN_WINDOW_S", 0.2)
    mic = Mock(device_name="Laptop Microphone", chunk_size=CHUNK, target_samplerate=16000)
    mic.devices.selectable_input_devices.return_value = devices

    def open_stream(_audio: Any, candidate: Dict[str, Any], _chunk: int) -> FakeStream:
        if candidate["index"] not in streams:
            raise OSError("device busy")
        return streams[candidate["index"]]

    monkeypatch.setattr(mic_scout, "open_input_stream", open_stream)
    on_heard, on_found = Mock(), Mock()
    return MicrophoneScout(mic, on_heard=on_heard, on_found=on_found), on_heard, on_found


def test_candidates_leave_out_the_current_device_and_loopback_inputs():
    devices = [{"index": 0, "name": "Laptop Microphone"}, HEADSET,
               {"index": 3, "name": "Stereomix (Realtek)"}, {"index": 4, "name": "CABLE Output (VB-Audio)"}]
    assert scout_devices(devices, "Laptop Microphone") == [ScoutDevice(2, "Headset", "Headset Microphone")]
    # The current device may be known by its full label only.
    assert scout_devices([HEADSET], "Headset Microphone") == []


def test_a_silent_default_switches_to_the_device_that_hears_the_user(monkeypatch):
    stream = FakeStream([LOUD] * 3)
    scout, on_heard, on_found = scout_with(monkeypatch, [HEADSET], {2: stream})
    scout._run()
    on_found.assert_called_once_with(ScoutDevice(2, "Headset", "Headset Microphone"))
    on_heard.assert_not_called()
    assert stream.closed


def test_a_working_default_is_confirmed_and_opens_no_other_device(monkeypatch):
    stream = FakeStream([LOUD] * 3)
    scout, on_heard, on_found = scout_with(monkeypatch, [HEADSET], {2: stream})
    for _ in range(mic_scout.MIN_SIGNAL_BLOCKS):
        scout.observe(pcm(LOUD))
    scout._run()
    on_heard.assert_called_once_with()
    on_found.assert_not_called()
    scout._mic.devices.selectable_input_devices.assert_not_called()


@pytest.mark.parametrize("peaks", [[FAINT] * 10, [LOUD] * (mic_scout.MIN_SIGNAL_BLOCKS - 1)],
                         ids=["background noise", "a short click"])
def test_noise_on_another_device_does_not_switch(monkeypatch, peaks):
    scout, on_heard, on_found = scout_with(monkeypatch, [HEADSET], {2: FakeStream(peaks)})
    scout._run()
    on_found.assert_not_called()
    on_heard.assert_not_called()


def test_another_device_must_clearly_beat_the_default(monkeypatch):
    """The default picks up the same voice, only a little quieter: it stays."""
    scout, _on_heard, on_found = scout_with(monkeypatch, [HEADSET], {2: FakeStream([LOUD] * 5)})
    scout.observe(pcm(LOUD // 2))
    scout._run()
    on_found.assert_not_called()


def test_a_device_that_cannot_be_opened_is_skipped(monkeypatch):
    other = {"index": 5, "name": "USB Mic"}
    scout, _on_heard, on_found = scout_with(monkeypatch, [HEADSET, other], {5: FakeStream([LOUD] * 3)})
    scout._run()
    on_found.assert_called_once_with(ScoutDevice(5, "USB Mic", "USB Mic"))


def test_a_stopped_scout_does_not_search(monkeypatch):
    scout, on_heard, on_found = scout_with(monkeypatch, [HEADSET], {2: FakeStream([LOUD] * 3)})
    scout.stop()
    scout._run()
    on_found.assert_not_called()
    on_heard.assert_not_called()
    scout._mic.devices.selectable_input_devices.assert_not_called()
