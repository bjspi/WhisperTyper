"""Confidence details: request fields per model, response condensing, delivery through transcribe()."""
from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.core import dsp
from app.core.models import transcription_form_fields
from app.core.timing import OperationTiming
from app.services import transcription as service
from app.services import transcription_diagnostics as diagnostics
from app.services.transcription import TranscriptionRequest

VERBOSE = {
    "text": " Gäste, wie du denkst.", "duration": 12.5,
    "segments": [
        {"start": 0.0, "end": 2.0, "text": " Gäste,", "avg_logprob": -1.5, "no_speech_prob": 0.01,
         "compression_ratio": 1.2, "temperature": 0.0, "tokens": [1, 2]},
        {"start": 2.0, "end": 10.0, "text": " wie du denkst.", "avg_logprob": -0.1, "no_speech_prob": 0.2,
         "compression_ratio": 1.4, "temperature": 0.0, "tokens": [3]},
    ],
    "words": [{"word": "Gäste,", "start": 0.5, "end": 1.0}, {"word": "wie", "start": 2.1, "end": 2.3}],
}
LOGPROBS = {
    "text": "Könntest du",
    "logprobs": [{"token": "K", "logprob": -0.05}, {"token": "önn", "logprob": -0.01},
                 {"token": "test", "logprob": 0.00004}, {"token": " du", "logprob": -0.5}],
}


def test_whisper_models_ask_for_verbose_json_with_word_timestamps():
    fields = transcription_form_fields("whisper-large-v3-turbo", "Hallo", 0.0, "de", diagnostics=True)
    assert fields["response_format"] == "verbose_json"
    assert fields["timestamp_granularities[]"] == ["word", "segment"]
    assert transcription_form_fields("whisper-1", "", 0.0, "de", diagnostics=True)["response_format"] == "verbose_json"


def test_gpt_transcription_models_ask_for_token_logprobs():
    fields = transcription_form_fields("gpt-4o-mini-transcribe", "Hallo", 0.0, "de", diagnostics=True)
    assert fields["include[]"] == "logprobs"
    assert "response_format" not in fields


def test_models_without_details_and_disabled_diagnostics_keep_the_plain_request():
    plain = transcription_form_fields("whisper-large-v3", "Hallo", 0.0, "de")
    assert "response_format" not in plain
    assert transcription_form_fields("my-whisper", "Hallo", 0.0, "de", diagnostics=True) == {
        "model": "my-whisper", "prompt": "Hallo", "temperature": 0.0, "language": "de"}
    assert transcription_form_fields("gpt-4o-transcribe-diarize", "", 0.0, "de", diagnostics=True)[
        "response_format"] == "json"


def test_verbose_summary_weights_segments_by_duration():
    details = diagnostics.extract(VERBOSE)
    summary = details["summary"]
    assert summary["avg_logprob"] == round((-1.5 * 2 + -0.1 * 8) / 10, 4)
    assert summary["min_segment_logprob"] == -1.5
    assert summary["max_no_speech_prob"] == 0.2
    assert summary["words"] == 2
    assert "tokens" not in details["segments"][0]
    assert details["words"][0] == {"word": "Gäste,", "start": 0.5, "end": 1.0}


def test_token_logprobs_are_grouped_into_words():
    details = diagnostics.extract(LOGPROBS)
    assert [w["word"] for w in details["words"]] == ["Könntest", "du"]
    assert details["words"][0]["logprob"] == -0.06
    assert details["words"][0]["tokens"] == 3
    assert details["summary"]["min_token_logprob"] == -0.5
    assert diagnostics.extract({"text": "plain"}) is None


def test_transcribe_reports_details_and_returns_the_text(tmp_path, monkeypatch):
    source = tmp_path / "recording.wav"
    dsp.write_wav(str(source), b"\x01\x00" * 1600, 16000)
    sent = {}

    def fake_request(method, url, **kwargs):
        sent.update(kwargs["data"])
        return Mock(status_code=200, json=lambda: VERBOSE, text="")

    monkeypatch.setattr(service, "request", fake_request)
    req = TranscriptionRequest("key", "https://api.groq.com/openai/v1/audio/transcriptions", str(source), "",
                               "whisper-large-v3-turbo", "de", 0.0, recording_format="wav", diagnostics=True)
    received = []
    text = service.transcribe(req, OperationTiming("recording"), lambda key, **_kw: key,
                              on_compressing=Mock(), on_transcribing=Mock(), on_diagnostics=received.append)
    assert text == VERBOSE["text"]
    assert sent["response_format"] == "verbose_json"
    assert received[0]["summary"]["segments"] == 2


def verbose(text, avg_logprob):
    return {"text": text, "duration": 5.0, "words": [],
            "segments": [{"start": 0.0, "end": 5.0, "text": text, "avg_logprob": avg_logprob,
                          "no_speech_prob": 0.0, "compression_ratio": 1.2}]}


def run_second_pass(tmp_path, monkeypatch, mode, first, second=None, second_delay=0.0):
    """Transcribe with a quality fallback; each response is a payload or an exception, told apart by prompt."""
    source = tmp_path / "recording.wav"
    dsp.write_wav(str(source), b"\x01\x00" * 1600, 16000)
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append(kwargs["data"])
        is_second = kwargs["data"]["prompt"] == ""
        if is_second:
            time.sleep(second_delay)
        response = second if is_second else first
        if isinstance(response, Exception):
            raise response
        return Mock(status_code=200, json=lambda: response, text="")

    monkeypatch.setattr(service, "request", fake_request)
    req = TranscriptionRequest("key", "https://api.groq.com/openai/v1/audio/transcriptions", str(source),
                               "Long prompt.", "whisper-large-v3-turbo", "de", 0.0, recording_format="wav",
                               second_pass=mode, second_pass_threshold=-0.5, second_pass_prompt="")
    received = []
    timing = OperationTiming("recording")
    text = service.transcribe(req, timing, lambda key, **_kw: key,
                              on_compressing=Mock(), on_transcribing=Mock(), on_diagnostics=received.append)
    return text, calls, received, timing


def test_a_confident_result_gets_no_second_transcription(tmp_path, monkeypatch):
    text, calls, received, _timing = run_second_pass(tmp_path, monkeypatch, "doubtful", verbose(" Alles da.", -0.1))
    assert text == " Alles da."
    assert len(calls) == 1
    assert calls[0]["response_format"] == "verbose_json"  # needed for the decision, even without diagnostics
    assert "second_pass" not in received[0]


def test_a_doubtful_result_is_transcribed_again_without_prompt_and_the_more_confident_text_wins(tmp_path, monkeypatch):
    text, calls, received, timing = run_second_pass(tmp_path, monkeypatch, "doubtful", verbose(" Gäste,", -0.99),
                                                    verbose(" Könntest du", -0.11))
    assert text == " Könntest du"
    assert [c["prompt"] for c in calls] == ["Long prompt.", ""]
    info = received[0]["second_pass"]
    assert info["chosen"] == "second"
    assert info["first"]["text"] == " Gäste,"
    assert received[0]["summary"]["avg_logprob"] == -0.11
    assert timing.notes == {"avg_logprob": -0.11, "second_pass": "second"}


def test_the_first_result_stays_when_the_second_is_less_confident_or_fails(tmp_path, monkeypatch):
    text, _calls, received, _timing = run_second_pass(tmp_path, monkeypatch, "doubtful", verbose(" Erster", -0.7),
                                                      verbose(" Zweiter", -0.9))
    assert text == " Erster"
    assert received[0]["second_pass"]["chosen"] == "first"
    text, _calls, received, timing = run_second_pass(tmp_path, monkeypatch, "doubtful", verbose(" Erster", -0.7),
                                                     RuntimeError("offline"))
    assert text == " Erster"
    assert received[0]["second_pass"]["error"] == "offline"
    assert timing.notes["second_pass"] == "failed"


def test_an_empty_result_gets_a_second_transcription(tmp_path, monkeypatch):
    empty = {"text": "", "duration": 5.0, "segments": [], "words": []}
    text, calls, _received, _timing = run_second_pass(tmp_path, monkeypatch, "doubtful", empty, verbose(" Text", -0.8))
    assert text == " Text"
    assert len(calls) == 2


def test_always_sends_both_and_does_not_wait_for_the_second_when_the_first_is_confident(tmp_path, monkeypatch):
    started = time.monotonic()
    text, calls, received, timing = run_second_pass(tmp_path, monkeypatch, "always", verbose(" Erster", -0.1),
                                                    verbose(" Zweiter", -0.05), second_delay=1.0)
    assert time.monotonic() - started < 0.8
    assert text == " Erster"
    assert received[0]["second_pass"]["second"] == "not awaited"
    assert timing.notes["second_pass"] == "first"
    time.sleep(1.1)
    assert [c["prompt"] for c in calls] == ["Long prompt.", ""] or [c["prompt"] for c in calls] == ["", "Long prompt."]


def test_always_takes_the_second_for_a_doubtful_first_and_falls_back_when_the_first_fails(tmp_path, monkeypatch):
    text, _calls, received, _timing = run_second_pass(tmp_path, monkeypatch, "always", verbose(" Gäste,", -0.99),
                                                      verbose(" Könntest du", -0.11), second_delay=0.2)
    assert text == " Könntest du"
    assert received[0]["second_pass"]["chosen"] == "second"
    text, _calls, received, _timing = run_second_pass(tmp_path, monkeypatch, "always", RuntimeError("timeout"),
                                                      verbose(" Könntest du", -0.11))
    assert text == " Könntest du"
    assert received[0]["second_pass"]["first"] == {"error": "timeout"}


def test_result_balloon_lists_the_notes_and_stays_longer():
    pytest.importorskip("PyQt6")
    from app.controllers.transcription import TranscriptionPipeline
    texts = {"transcription_done_message": "Done", "result_note_logprob": "avg {value}",
             "result_note_second_pass_second": "2nd used", "result_note_hedge_won": "2nd connection"}
    shown = []
    fake = SimpleNamespace(_ctx=SimpleNamespace(tr=lambda key, **kw: texts[key].format(**kw)),
                           _config={"transcription_second_pass_logprob": -0.5},
                           _notifier=SimpleNamespace(show=lambda message, ms, check, detail="", detail_level="":
                                                     shown.append((message, ms, detail, detail_level))))

    def show(**notes):
        timing = OperationTiming("recording")
        timing.notes.update(notes)
        TranscriptionPipeline._show_done(fake, "transcription_done_message", 1000, timing)
        return shown[-1]

    assert show() == ("Done", 1000, "", "")
    assert show(avg_logprob=-0.1234, second_pass="second", hedge_winner=2) == (
        "Done", 1440, "avg -0.12 · 2nd used · 2nd connection", "warn")
    assert show(avg_logprob=-0.2)[3] == "good"
    assert show(avg_logprob=-0.4)[3] == "warn"
    assert show(avg_logprob=-0.6)[3] == "bad"
