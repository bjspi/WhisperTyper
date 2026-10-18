"""Confidence details: request fields per model, response condensing, delivery through transcribe()."""
from __future__ import annotations

from unittest.mock import Mock

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
