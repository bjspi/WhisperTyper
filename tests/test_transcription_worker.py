"""Recording format uploads preserve originals and never invoke FFmpeg."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

pytest.importorskip("PyQt6.QtCore")

from app.core import dsp
from app.core.timing import OperationTiming
from app.services import transcription_worker as service


@pytest.mark.parametrize("recording_format", ["wav", "aac"])
def test_recording_upload_uses_selected_format_and_cleans_temp(monkeypatch, tmp_path, recording_format):
    source = tmp_path / "recording.wav"
    dsp.write_wav(str(source), b"\x01\x00" * 16000, 16000)
    original = source.read_bytes()
    worker = service.TranscriptionWorker("test-key", "https://example.invalid/transcribe", str(source), "", "whisper", "en", 0,
                                         timing=OperationTiming("recording"), recording_format=recording_format)
    prepared = []

    def encode(source_path, destination, bitrate):
        assert source_path == str(source)
        assert bitrate == 64
        Path(destination).write_bytes(b"encoded-aac")
        prepared.append(destination)

    def request(_method, _url, **kwargs):
        filename, handle, content_type = kwargs["files"]["file"]
        if recording_format == "aac":
            assert filename.endswith(".m4a")
            assert content_type == "audio/mp4"
            assert handle.read() == b"encoded-aac"
        else:
            assert filename.endswith(".wav")
            assert handle.read() == original
        return Mock(status_code=200, json=lambda: {"text": "result"})

    monkeypatch.setattr(service, "encode_wav_to_aac", encode)
    monkeypatch.setattr(service, "request", request)
    prepare = Mock(side_effect=AssertionError("Recordings must not invoke FFmpeg"))
    monkeypatch.setattr(service.ffmpeg, "prepare_upload", prepare)
    finished, errors = [], []
    worker.finished.connect(finished.append)
    worker.error.connect(lambda *args: errors.append(args))
    worker.run()
    assert finished == ["result"]
    assert errors == []
    assert source.read_bytes() == original
    assert all(not Path(path).exists() for path in prepared)
    assert ("audio_encode_end" in worker.timing._events) == (recording_format == "aac")
    assert not prepare.called


@pytest.mark.parametrize("failure", ["encoder", "size", "http"])
def test_recording_failures_retain_wav_and_clean_partial_aac(monkeypatch, tmp_path, failure):
    source = tmp_path / "recording.wav"
    dsp.write_wav(str(source), b"\x01\x00" * 16000, 16000)
    original = source.read_bytes()
    prepared = []

    def encode(_source, destination, _bitrate):
        prepared.append(destination)
        Path(destination).write_bytes(b"partial-aac")
        if failure == "encoder":
            raise OSError("Simulated native encoder failure")

    monkeypatch.setattr(service, "encode_wav_to_aac", encode)
    request = Mock(return_value=Mock(status_code=500, text="Simulated HTTP failure"))
    monkeypatch.setattr(service, "request", request)
    prepare = Mock(side_effect=AssertionError("No FFmpeg"))
    monkeypatch.setattr(service.ffmpeg, "prepare_upload", prepare)
    worker = service.TranscriptionWorker("test-key", "https://example.invalid/transcribe", str(source), "", "whisper", "en", 0,
                                         recording_format="aac", max_upload_bytes=1 if failure == "size" else 1024 * 1024)
    errors = []
    worker.error.connect(lambda *args: errors.append(args))
    worker.run()
    assert len(errors) == 1
    assert errors[0][1] == str(source)
    assert source.read_bytes() == original
    assert all(not Path(path).exists() for path in prepared)
    assert request.called == (failure == "http")
    assert not prepare.called
