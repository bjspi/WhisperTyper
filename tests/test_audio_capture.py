"""Recording-stop races with fake input streams; no microphone or sounds are used."""
from __future__ import annotations

import threading
import time
from collections import deque

import pytest

pytest.importorskip("pyaudio")  # Exercised by the Windows audio/clipboard job, without native devices.

from app.core.timing import OperationTiming
from app.mixins.audio_mixin import AudioMixin


class Stream:
    def __init__(self, owner, fail=False, drain_fail=False):
        """Block one fake read so tests can stop at the exact race boundary."""
        self.owner = owner
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
            self.owner.audio_capture_running = False
            if self.fail:
                raise OSError("synthetic driver failure")
            return b"\x01\x00" * frames
        assert frames == 80
        return b"\x02\x00" * frames

    def get_read_available(self):
        if self.drain_fail:
            raise OSError("synthetic drain failure")
        return 80


class CaptureHarness(AudioMixin):
    def __init__(self, fail=False, drain_fail=False):
        """Initialize only capture state, with a fake stream and no native device."""
        self.audio_state_lock = threading.Lock()
        self._background_read_pending = False
        self._background_stop = None
        self.is_recording = True
        self.audio_capture_running = True
        self.audio_capture_thread = None
        self.config = {"windows_keep_mic_hot_idle_minutes": 15}
        self.last_transcription_activity_ts = time.monotonic()
        self.recorded_frames = [b"\x03\x00" * 1024]
        self.pre_record_buffer = deque(maxlen=12)
        self.chunk_size = 1024
        self.current_input_samplerate = 16000
        self.stream = Stream(self, fail=fail, drain_fail=drain_fail)

    def _ensure_input_stream(self):
        return self.stream

    def _close_input_stream(self):
        pass

    def _use_windows_keep_mic_hot(self):
        return True


@pytest.mark.parametrize("drain_fail", [False, True])
def test_stop_keeps_the_inflight_block_and_drains_only_available_samples(drain_fail):
    recorder = CaptureHarness(drain_fail=drain_fail)
    reader = threading.Thread(target=recorder._audio_capture_loop, daemon=True)
    reader.start()
    try:
        assert recorder.stream.entered.wait(1)
        timing = OperationTiming("recording")
        done = recorder._begin_recording_stop(timing)
        assert done is not None and not recorder.is_recording
        assert not done.is_set()
        recorder.stream.release.set()
        recorder._wait_for_recording_tail(done, timing)
        reader.join(1)
        assert not reader.is_alive()
        assert recorder.recorded_frames[:2] == [b"\x03\x00" * 1024, b"\x01\x00" * 1024]
        assert recorder.stream.calls == ([1024] if drain_fail else [1024, 80])
        if not drain_fail:
            assert recorder.recorded_frames[2] == b"\x02\x00" * 80
        else:
            assert "recording_tail_drain_failed" in timing._events
        assert "recording_tail_saved" in timing._events
        assert "recording_tail_timeout" not in timing._events
        assert recorder._background_stop is None
    finally:
        recorder.stream.release.set()
        reader.join(2)


def test_stop_without_an_active_read_has_no_wait():
    recorder = CaptureHarness()
    timing = OperationTiming("recording")
    assert recorder._begin_recording_stop(timing) is None
    assert not recorder.is_recording
    assert "recording_tail_wait_start" not in timing._events


def test_timeout_cannot_modify_an_already_saved_recording():
    recorder = CaptureHarness()
    reader = threading.Thread(target=recorder._audio_capture_loop, daemon=True)
    reader.start()
    try:
        assert recorder.stream.entered.wait(1)
        timing = OperationTiming("recording")
        done = recorder._begin_recording_stop(timing)
        recorder._wait_for_recording_tail(done, timing)
        assert "recording_tail_timeout" in timing._events
        snapshot = list(recorder.recorded_frames)
        recorder.stream.release.set()
        reader.join(1)
        assert recorder.recorded_frames == snapshot
    finally:
        recorder.stream.release.set()
        reader.join(2)


def test_failed_input_read_releases_the_stop_without_claiming_saved_audio():
    recorder = CaptureHarness(fail=True)
    reader = threading.Thread(target=recorder._audio_capture_loop, daemon=True)
    reader.start()
    try:
        assert recorder.stream.entered.wait(1)
        timing = OperationTiming("recording")
        done = recorder._begin_recording_stop(timing)
        recorder.stream.release.set()
        recorder._wait_for_recording_tail(done, timing)
        reader.join(1)
        assert "recording_tail_read_failed" in timing._events
        assert "recording_tail_saved" not in timing._events
        assert "recording_tail_timeout" not in timing._events
        assert len(recorder.recorded_frames) == 1
    finally:
        recorder.stream.release.set()
        reader.join(2)


def test_cancel_does_not_attach_a_tail_or_save_pending_audio():
    recorder = CaptureHarness()
    reader = threading.Thread(target=recorder._audio_capture_loop, daemon=True)
    reader.start()
    try:
        assert recorder.stream.entered.wait(1)
        with recorder.audio_state_lock:
            recorder.is_recording = False
        recorder.stream.release.set()
        reader.join(1)
        assert len(recorder.recorded_frames) == 1
    finally:
        recorder.stream.release.set()
        reader.join(2)
