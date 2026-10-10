"""Hedged transcription: a stalled request is raced by a second one over a fresh connection."""
from __future__ import annotations

import threading
import time
from unittest.mock import Mock

import pytest

from app.core import dsp
from app.core.timing import OperationTiming
from app.services import transcription as service
from app.services.transcription import TranscriptionError, TranscriptionRequest


@pytest.fixture(autouse=True)
def quick_thresholds(monkeypatch):
    monkeypatch.setattr(service, "_HEDGE_UPLOAD_MIN_S", 0.05)
    monkeypatch.setattr(service, "_HEDGE_RESPONSE_MIN_S", 0.05)


def ok(text):
    return Mock(status_code=200, json=lambda: {"text": text}, text="")


class FakeApi:
    """Scripted attempts: each entry is (delay before upload is done, delay before answer, response)."""

    def __init__(self, *script):
        """Store the per-attempt script and record every call."""
        self.script = list(script)
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, method, url, **kwargs):
        with self.lock:
            index = len(self.calls)
            self.calls.append(kwargs)
        upload_delay, answer_delay, response = self.script[index]
        on_phase = kwargs.get("on_phase")
        time.sleep(upload_delay)
        if on_phase:
            on_phase("request_sent")
        time.sleep(answer_delay)
        if on_phase:
            on_phase("first_byte")
        return response


def run(tmp_path, monkeypatch, api, hedge_key="key-two"):
    source = tmp_path / "recording.wav"
    dsp.write_wav(str(source), b"\x01\x00" * 1600, 16000)
    monkeypatch.setattr(service, "request", api)
    req = TranscriptionRequest("key-one", "https://api.groq.com/openai/v1/audio/transcriptions", str(source),
                               "", "whisper-large-v3", "de", 0.0, recording_format="wav", hedge_api_key=hedge_key)
    timing = OperationTiming("recording")
    text = service.transcribe(req, timing, lambda key, **_kw: key, on_compressing=Mock(), on_transcribing=Mock())
    return text, timing


def test_stalled_upload_is_raced_by_a_fresh_connection_with_the_next_key(tmp_path, monkeypatch):
    api = FakeApi((1.0, 0.0, ok("slow")), (0.0, 0.0, ok("fast")))
    text, timing = run(tmp_path, monkeypatch, api)
    assert text == "fast"
    first, second = api.calls
    assert not first.get("fresh_connection") and second["fresh_connection"] is True
    assert second["headers"]["Authorization"] == "Bearer key-two"
    assert "transcription_hedge_won_2" in timing._events


def test_stalled_response_after_a_quick_upload_also_triggers(tmp_path, monkeypatch):
    api = FakeApi((0.0, 1.0, ok("slow")), (0.0, 0.0, ok("fast")))
    text, _timing = run(tmp_path, monkeypatch, api)
    assert text == "fast" and len(api.calls) == 2


def test_fast_first_attempt_sends_no_second_request(tmp_path, monkeypatch):
    api = FakeApi((0.0, 0.0, ok("only")))
    text, timing = run(tmp_path, monkeypatch, api)
    assert text == "only" and len(api.calls) == 1
    assert "transcription_hedge_started" not in timing._events


def test_fast_failure_is_reported_as_without_hedging(tmp_path, monkeypatch):
    api = FakeApi((0.0, 0.0, Mock(status_code=500, text="boom")))
    with pytest.raises(TranscriptionError):
        run(tmp_path, monkeypatch, api)
    assert len(api.calls) == 1


def test_a_failed_attempt_waits_for_the_other_and_both_failing_raises(tmp_path, monkeypatch):
    api = FakeApi((1.0, 0.0, ok("late but fine")), (0.0, 0.0, Mock(status_code=503, text="busy")))
    text, timing = run(tmp_path, monkeypatch, api)
    assert text == "late but fine" and "transcription_hedge_won_1" in timing._events

    api = FakeApi((0.3, 0.0, Mock(status_code=500, text="a")), (0.0, 0.0, Mock(status_code=500, text="b")))
    with pytest.raises(TranscriptionError):
        run(tmp_path, monkeypatch, api)


def test_without_a_hedge_key_there_is_never_a_second_request(tmp_path, monkeypatch):
    api = FakeApi((0.3, 0.0, ok("patient")))
    text, _timing = run(tmp_path, monkeypatch, api, hedge_key=None)
    assert text == "patient" and len(api.calls) == 1
    assert "on_phase" not in api.calls[0]


def test_thresholds_grow_with_the_upload_size():
    small_upload, small_wait = service.hedge_thresholds(50_000)
    large_upload, large_wait = service.hedge_thresholds(6 * 1024 * 1024)
    assert small_upload == service._HEDGE_UPLOAD_MIN_S and large_upload > small_upload
    assert large_wait > small_wait >= service._HEDGE_RESPONSE_MIN_S


def test_upload_finishing_late_starts_the_response_deadline_instead_of_hedging(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "_HEDGE_RESPONSE_MIN_S", 0.3)
    # Upload done within its 0.05 s budget, answer after 0.15 s: past the upload budget, but within
    # the 0.3 s response wait that starts when the upload is done.
    api = FakeApi((0.01, 0.15, ok("first")), (0.0, 0.0, ok("second")))
    text, timing = run(tmp_path, monkeypatch, api)
    assert text == "first"
    assert len(api.calls) == 1
    assert "transcription_hedge_started" not in timing._events
