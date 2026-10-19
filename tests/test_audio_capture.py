"""Recording-stop races of HotMicCapture with a fake input stream; no microphone is used."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("pyaudio")  # Exercised by the Windows audio/clipboard job, without native devices.

from app.audio import capture as capture_module
from app.audio.capture import HotMicCapture, MicrophoneInput, OnDemandCapture
from app.core.timing import OperationTiming

PREROLL = b"\x03\x00" * 1024


class Stream:
    def __init__(self, capture_ref, fail=False, drain_fail=False):
        """Block the first read so tests can stop at the exact race boundary."""
        self.capture_ref = capture_ref
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = []
        self.fail = fail
        self.drain_fail = drain_fail

    def read(self, frames, **_kwargs):
        self.calls.append(frames)
        if len(self.calls) == 1:
            self.entered.set()
            assert self.release.wait(2)
            self.capture_ref[0]._running = False  # End the reader loop after this read.
            if self.fail:
                raise OSError("synthetic driver failure")
            return b"\x01\x00" * frames
        assert frames == 80
        return b"\x02\x00" * frames

    def get_read_available(self):
        if self.drain_fail:
            raise OSError("synthetic drain failure")
        return 80


class Microphone:
    """The MicrophoneInput surface the captures use, without PyAudio."""

    target_samplerate = 16000
    samplerate = 16000
    chunk_size = 1024
    device_name = ""

    def __init__(self, stream):
        """Hand out the given fake stream."""
        self.stream = stream
        self.closed = 0

    def open(self):
        return self.stream

    def close(self):
        self.closed += 1


def recording(fail=False, drain_fail=False):
    """A capture recording since the pre-roll, whose reader is blocked inside its first read."""
    capture_ref: list = []
    stream = Stream(capture_ref, fail=fail, drain_fail=drain_fail)
    capture = HotMicCapture(Microphone(stream), idle_expired=lambda: False, on_block=lambda _data: None)
    capture_ref.append(capture)
    capture._preroll.append(PREROLL)
    capture.start_recording()
    capture.start()
    assert stream.entered.wait(1)
    return capture, stream


def finish(capture, stream):
    stream.release.set()
    if capture._thread:
        capture._thread.join(2)


@pytest.mark.parametrize("drain_fail", [False, True])
def test_stop_keeps_the_inflight_block_and_drains_only_available_samples(drain_fail):
    capture, stream = recording(drain_fail=drain_fail)
    try:
        timing = OperationTiming("recording")
        collect = capture.stop_recording(timing)
        stream.release.set()
        frames = collect()
        capture._thread.join(1)
        assert frames[:2] == [PREROLL, b"\x01\x00" * 1024]
        assert stream.calls == ([1024] if drain_fail else [1024, 80])
        if drain_fail:
            assert "recording_tail_drain_failed" in timing._events
        else:
            assert frames[2] == b"\x02\x00" * 80
        assert "recording_tail_saved" in timing._events
        assert "recording_tail_timeout" not in timing._events
        assert capture._pending_stop is None
    finally:
        finish(capture, stream)


def test_stop_without_an_active_read_has_no_wait():
    capture = HotMicCapture(Microphone(None), idle_expired=lambda: False, on_block=lambda _data: None)
    capture.start_recording()
    timing = OperationTiming("recording")
    assert capture.stop_recording(timing)() == []
    assert "recording_tail_wait_start" not in timing._events


def test_timeout_cannot_modify_an_already_returned_recording():
    capture, stream = recording()
    try:
        timing = OperationTiming("recording")
        frames = capture.stop_recording(timing)()
        assert "recording_tail_timeout" in timing._events
        stream.release.set()
        capture._thread.join(1)
        assert capture.stop_recording(OperationTiming("recording"))() == frames == [PREROLL]
    finally:
        finish(capture, stream)


def test_failed_input_read_releases_the_stop_without_claiming_saved_audio():
    capture, stream = recording(fail=True)
    try:
        timing = OperationTiming("recording")
        collect = capture.stop_recording(timing)
        stream.release.set()
        frames = collect()
        capture._thread.join(1)
        assert "recording_tail_read_failed" in timing._events
        assert "recording_tail_saved" not in timing._events
        assert "recording_tail_timeout" not in timing._events
        assert frames == [PREROLL]
    finally:
        finish(capture, stream)


def test_cancel_does_not_attach_a_tail_or_save_pending_audio():
    capture, stream = recording()
    try:
        capture.cancel_recording()
        stream.release.set()
        capture._thread.join(1)
        assert capture._frames == [PREROLL]
    finally:
        finish(capture, stream)


class CountingStream:
    """Delivers numbered blocks; the test decides when the capture stops reading."""

    def __init__(self):
        """Start counting from the first block."""
        self.blocks = 0

    def read(self, frames, **_kwargs):
        self.blocks += 1
        time.sleep(0.002)
        return bytes([min(self.blocks, 255), 0]) * frames

    def get_read_available(self):
        return 0


def wait_for(condition):
    deadline = time.monotonic() + 2
    while not condition():
        assert time.monotonic() < deadline
        time.sleep(0.002)


@pytest.mark.parametrize("hot", [True, False])
def test_first_recorded_block_is_announced_once_and_restart_discards_the_take(hot):
    announced = threading.Event()
    calls: list[None] = []

    def on_audio():
        calls.append(None)
        announced.set()

    stream = CountingStream()
    if hot:
        capture = HotMicCapture(Microphone(stream), idle_expired=lambda: False, on_block=lambda _data: None,
                                on_recording_audio=on_audio)
        capture._preroll.append(PREROLL)
        capture.start()
        wait_for(lambda: stream.blocks > 0)  # The warm microphone reads before the recording starts.
        assert calls == []
        capture.start_recording()
    else:
        capture = OnDemandCapture(Microphone(stream), on_block=lambda _data: None, on_failure=lambda: None,
                                  on_recording_audio=on_audio)
        capture.start_recording()
    assert announced.wait(2)
    wait_for(lambda: len(capture._frames) > 3)

    restarted_at = stream.blocks  # Only the block being read right now may still end up in the take.
    capture.restart_recording()
    wait_for(lambda: len(capture._frames) > 2)
    frames = capture.stop_recording(OperationTiming("recording"))()
    if hot:
        capture.stop()

    assert calls == [None]  # A restart keeps recording; the take is not announced again.
    assert PREROLL not in frames  # Nothing from before the restart, not even the pre-roll.
    assert frames and frames[0][0] >= restarted_at


def capture_recording(hot, stream):
    """A running capture of numbered blocks that already holds a few of them."""
    if hot:
        capture = HotMicCapture(Microphone(stream), idle_expired=lambda: False, on_block=lambda _data: None)
        capture.start()
        capture.start_recording()
    else:
        capture = OnDemandCapture(Microphone(stream), on_block=lambda _data: None, on_failure=lambda: None)
        capture.start_recording()
    wait_for(lambda: len(capture._frames) > 2)
    return capture


@pytest.mark.parametrize("hot", [True, False])
def test_paused_blocks_are_dropped_and_resuming_continues_the_take(hot):
    stream = CountingStream()
    capture = capture_recording(hot, stream)
    capture.set_paused(True)
    paused_at = stream.blocks  # The block being read right now may still be kept.
    wait_for(lambda: stream.blocks > paused_at + 5)
    resumed_at = stream.blocks  # The block being read when resuming may already count again.
    capture.set_paused(False)
    wait_for(lambda: stream.blocks > resumed_at + 3)
    frames = capture.stop_recording(OperationTiming("recording"))()
    if hot:
        capture.stop()
    numbers = [frame[0] for frame in frames]
    assert not [n for n in numbers if paused_at + 1 < n < resumed_at]
    assert numbers[0] <= paused_at and numbers[-1] > resumed_at


@pytest.mark.parametrize("hot", [True, False])
def test_stopping_while_paused_sends_the_audio_before_the_pause(hot):
    stream = CountingStream()
    capture = capture_recording(hot, stream)
    capture.set_paused(True)
    paused_at = stream.blocks
    wait_for(lambda: stream.blocks > paused_at + 5)
    frames = capture.stop_recording(OperationTiming("recording"))()
    if hot:
        capture.stop()
    assert frames and all(frame[0] <= paused_at + 1 for frame in frames)


@pytest.mark.parametrize(("connected", "notified"), [(False, []), (True, ["Headset"])])
def test_device_fallback_is_reported_only_when_a_connected_device_fails(monkeypatch, connected, notified):
    """A device that is simply not plugged in falls back quietly; a connected one that fails is reported once."""
    reports: list[str] = []
    mic = MicrophoneInput(lambda: {"input_device_name": "Headset"}, 16000, 1024, reports.append)
    default = {"index": None, "name": "default input", "channels": 1, "samplerate": 16000}
    configured = {"index": 3, "name": "Headset", "channels": 1, "samplerate": 16000}
    candidates = [configured, default] if connected else [default]
    monkeypatch.setattr(mic, "pyaudio", lambda refresh=False: object())
    monkeypatch.setattr(mic.devices, "preferred_input_stream_candidates", lambda _audio: candidates)

    def open_stream(_audio, candidate, _chunk):
        if candidate["index"] is not None:
            raise OSError("device busy")
        return SimpleNamespace(stop_stream=lambda: None, close=lambda: None)

    monkeypatch.setattr(capture_module, "_open_input_stream", open_stream)
    for _ in range(2):
        mic.open()
        mic.close()
    assert reports == notified


def test_stop_leaves_the_stream_to_a_reader_stuck_in_a_read(monkeypatch):
    monkeypatch.setattr(capture_module, "_JOIN_WAIT_S", 0.05)
    capture, stream = recording()
    try:
        assert capture.stop() is False  # The reader never returned from its first read.
        assert capture.running and capture._mic.closed == 0  # Closing under the read would crash PortAudio.
        stream.release.set()
        capture._thread.join(1)
        assert not capture.running and capture._mic.closed == 1  # The reader closes it once the read returns.
        assert capture.stop() is True
    finally:
        finish(capture, stream)


def test_on_demand_collect_returns_the_audio_so_far_while_the_reader_hangs(monkeypatch):
    monkeypatch.setattr(capture_module, "_JOIN_WAIT_S", 0.05)
    stream = Stream([SimpleNamespace()])
    capture = OnDemandCapture(Microphone(stream), on_block=lambda _data: None, on_failure=lambda: None)
    capture.start_recording()
    assert stream.entered.wait(1)
    collect = capture.stop_recording(OperationTiming("recording"))
    started = time.monotonic()
    assert collect() == []
    assert time.monotonic() - started < 1 and capture._thread.is_alive()
    stream.release.set()
    capture._thread.join(1)
    assert not capture._thread.is_alive()


class FailingStream:
    """A stream whose reads fail, so the reader closes it and opens the microphone again."""

    def read(self, _frames, **_kwargs):
        raise OSError("synthetic read failure")


class UnopenableMicrophone(Microphone):
    """Cannot be opened, except for one open that hands out a stream that fails to read."""

    def __init__(self, opens_on):
        """Succeed on the ``opens_on``-th open only."""
        super().__init__(FailingStream())
        self.opens = 0
        self.opens_on = opens_on

    def open(self):
        self.opens += 1
        if self.opens != self.opens_on:
            raise OSError("no input device")
        return self.stream


class RecordedWake:
    """Stands in for the reader's pause between open attempts: records the delays instead of waiting."""

    def __init__(self):
        """Start with no pauses recorded."""
        self.delays: list[float] = []

    def clear(self):
        pass

    def set(self):
        pass

    def wait(self, delay):
        self.delays.append(delay)
        time.sleep(0.001)
        return False


def test_open_failures_back_off_and_a_recording_is_reported_once(monkeypatch):
    monkeypatch.setattr(capture_module, "_READ_RETRY_S", 0.01)
    monkeypatch.setattr(capture_module, "_OPEN_RETRY_MAX_S", 0.04)
    failures: list[None] = []
    capture = HotMicCapture(UnopenableMicrophone(opens_on=3), idle_expired=lambda: False,
                            on_block=lambda _data: None, on_failure=lambda: failures.append(None))
    wake = RecordedWake()
    capture._wake = wake
    capture.start()
    try:
        wait_for(lambda: len(wake.delays) >= 6)
        assert failures == []  # Nothing to report while no recording runs.
        capture.start_recording()
        wait_for(lambda: failures)
        wait_for(lambda: len(wake.delays) >= 10)
        assert failures == [None]
        capture.cancel_recording()
    finally:
        capture.stop()
    # The third open succeeded and its read failed: the pause starts over from the shortest one.
    assert wake.delays[:6] == [0.01, 0.02, 0.01, 0.02, 0.04, 0.04]


def test_preroll_is_sized_for_the_rate_the_stream_opened_at():
    stream = CountingStream()
    microphone = Microphone(stream)
    microphone.samplerate = 48000  # The device's native rate; the app rate stays 16 kHz.
    capture = HotMicCapture(microphone, idle_expired=lambda: False, on_block=lambda _data: None)
    assert capture._preroll.maxlen == 11
    capture.start()
    try:
        wait_for(lambda: stream.blocks > 0)
    finally:
        capture.stop()
    assert capture._preroll.maxlen == 35 and len(capture._preroll) >= 1


def test_a_device_name_cut_by_mme_with_a_trailing_space_is_still_found(monkeypatch):
    """MME truncates names at 31 characters, often ending in a space; the stored name is stripped."""
    from app.audio import device_selector as selector_module

    monkeypatch.setattr(selector_module, "is_MACOS", False)
    mme_name = "Mikrofonarray (2- Intel® Smart "  # 31 characters, as MME reports it
    infos = {
        0: {"index": 0, "name": "Microsoft Soundmapper - Input", "maxInputChannels": 2, "hostApi": 0},
        1: {"index": 1, "name": mme_name, "maxInputChannels": 2, "hostApi": 0, "defaultSampleRate": 48000.0},
    }
    audio = SimpleNamespace(
        get_default_host_api_info=lambda: {"index": 0},
        get_device_count=lambda: len(infos),
        get_device_info_by_index=lambda index: infos[index],
        get_default_input_device_info=lambda: infos[1],
    )
    config = {"input_device_name": ""}
    selector = selector_module.InputDeviceSelector(lambda: config, 16000, lambda: audio)
    devices = selector.selectable_input_devices()
    assert devices == [{"index": 1, "name": mme_name.strip()}]  # what the menus offer and store

    config["input_device_name"] = devices[0]["name"]
    candidates = selector.preferred_input_stream_candidates(audio)
    assert [c["index"] for c in candidates] == [1, 1, None]  # the chosen device first, at 16 kHz and natively
    assert capture_module._default_input_name(audio) == mme_name.strip()
