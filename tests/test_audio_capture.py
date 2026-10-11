"""Recording-stop races of HotMicCapture with a fake input stream; no microphone is used."""
from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("pyaudio")  # Exercised by the Windows audio/clipboard job, without native devices.

from app.audio.capture import HotMicCapture, OnDemandCapture
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

    def open(self):
        return self.stream

    def close(self):
        pass


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
