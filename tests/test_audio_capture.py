"""Recording-stop races of HotMicCapture with a fake input stream; no microphone is used."""
from __future__ import annotations

import threading

import pytest

pytest.importorskip("pyaudio")  # Exercised by the Windows audio/clipboard job, without native devices.

from app.audio.capture import HotMicCapture
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
    """The MicrophoneInput surface HotMicCapture uses, without PyAudio."""

    target_samplerate = 16000
    samplerate = 16000
    chunk_size = 1024

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
