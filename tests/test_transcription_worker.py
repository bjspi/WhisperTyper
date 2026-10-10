"""Recording format uploads preserve originals and use FFmpeg only above the upload limit."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

pytest.importorskip("PyQt6.QtCore")

from app.core import dsp
from app.core.timing import OperationTiming
from app.services import transcription as service
from app.services.transcription import TranscriptionRequest
from app.services.transcription_worker import TranscriptionWorker


def make_worker(source, *, timing=None, tr=None, **request):
    """A worker for ``source`` with fixed credentials/model and per-test request options."""
    return TranscriptionWorker(
        TranscriptionRequest("test-key", "https://example.invalid/transcribe", str(source), "", "whisper", "en", 0, **request),
        timing=timing, tr=tr,
    )


@pytest.mark.parametrize("recording_format", ["wav", "aac"])
def test_recording_upload_uses_selected_format_and_cleans_temp(monkeypatch, tmp_path, recording_format):
    source = tmp_path / "recording.wav"
    dsp.write_wav(str(source), b"\x01\x00" * 16000, 16000)
    original = source.read_bytes()
    worker = make_worker(source, timing=OperationTiming("recording"), recording_format=recording_format)
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
    monkeypatch.setattr(service, "available_aac_bitrates", lambda: (64,))
    monkeypatch.setattr(service, "request", request)
    prepare = Mock(side_effect=AssertionError("Recordings within the limit must not invoke FFmpeg"))
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
            raise OSError("Simulated AAC encoder failure")

    monkeypatch.setattr(service, "encode_wav_to_aac", encode)
    monkeypatch.setattr(service, "available_aac_bitrates", lambda: (64,))
    request = Mock(return_value=Mock(status_code=500, text="Simulated HTTP failure"))
    monkeypatch.setattr(service, "request", request)
    prepare = Mock(side_effect=AssertionError("No FFmpeg"))
    monkeypatch.setattr(service.ffmpeg, "prepare_upload", prepare)
    # An oversized recording without an installed FFmpeg is rejected with a translated hint.
    monkeypatch.setattr(service.ffmpeg, "resolve_ffmpeg", lambda _setting: None)
    worker = make_worker(source, recording_format="aac", max_upload_bytes=1 if failure == "size" else 1024 * 1024)
    errors = []
    worker.error.connect(lambda *args: errors.append(args))
    worker.run()
    assert len(errors) == 1
    assert errors[0][1] == str(source)
    assert source.read_bytes() == original
    assert all(not Path(path).exists() for path in prepared)
    assert request.called == (failure == "http")
    assert not prepare.called


@pytest.mark.parametrize("recording_format", ["wav", "aac"])
def test_oversized_recording_falls_back_to_ffmpeg_from_retained_wav(monkeypatch, tmp_path, recording_format):
    source = tmp_path / "recording.wav"
    dsp.write_wav(str(source), b"\x01\x00" * 16000, 16000)
    original = source.read_bytes()
    encoded = []

    def encode(_source, destination, _bitrate):
        Path(destination).write_bytes(b"too-large-aac")
        encoded.append(destination)

    compressed = tmp_path / "compressed.mp3"

    def prepare(exe, src_path, *, transcode_source, max_bytes, min_bitrate_kbps):
        assert (exe, src_path, transcode_source) == ("ffmpeg-bin", str(source), False)
        assert all(not Path(path).exists() for path in encoded)  # The AAC is replaced, not transcoded.
        compressed.write_bytes(b"x")
        return str(compressed), str(compressed)

    def request(_method, _url, **kwargs):
        filename, handle, _content_type = kwargs["files"]["file"]
        assert filename == "compressed.mp3" and handle.read() == b"x"
        return Mock(status_code=200, json=lambda: {"text": "result"})

    monkeypatch.setattr(service, "encode_wav_to_aac", encode)
    monkeypatch.setattr(service, "available_aac_bitrates", lambda: (64,))
    monkeypatch.setattr(service.ffmpeg, "resolve_ffmpeg", lambda setting: "ffmpeg-bin" if setting == "configured" else None)
    monkeypatch.setattr(service.ffmpeg, "prepare_upload", prepare)
    monkeypatch.setattr(service, "request", request)
    worker = make_worker(source, recording_format=recording_format, max_upload_bytes=5, ffmpeg_setting="configured")
    finished, compressing = [], []
    worker.finished.connect(finished.append)
    worker.compressing.connect(compressing.append)
    worker.run()
    assert finished == ["result"]
    assert compressing == ["recording.wav"]
    assert "audio_compress_end" in worker.timing._events
    assert source.read_bytes() == original
    assert not compressed.exists()


def test_worker_errors_are_translated(monkeypatch, tmp_path):
    source = tmp_path / "recording.wav"
    dsp.write_wav(str(source), b"\x01\x00" * 16000, 16000)
    monkeypatch.setattr(service, "request", Mock(return_value=Mock(status_code=401, text="denied")))
    worker = make_worker(source, recording_format="wav", tr=lambda key, **kwargs: f"<{key}:{sorted(kwargs.items())}>")
    errors = []
    worker.error.connect(lambda *args: errors.append(args))
    worker.run()
    assert errors == [("<transcription_api_error:[('details', 'denied'), ('status', 401)]>", str(source))]


def test_aac_choice_without_pyav_uploads_wav_instead_of_failing(monkeypatch, tmp_path):
    source = tmp_path / "recording.wav"
    dsp.write_wav(str(source), b"\x01\x00" * 16000, 16000)
    original = source.read_bytes()

    def request(_method, _url, **kwargs):
        filename, handle, _content_type = kwargs["files"]["file"]
        assert filename == "recording.wav" and handle.read() == original
        return Mock(status_code=200, json=lambda: {"text": "result"})

    monkeypatch.setattr(service, "available_aac_bitrates", lambda: ())
    monkeypatch.setattr(service, "encode_wav_to_aac", Mock(side_effect=AssertionError("PyAV is not installed")))
    monkeypatch.setattr(service, "request", request)
    worker = make_worker(source, recording_format="aac")
    finished = []
    worker.finished.connect(finished.append)
    worker.run()
    assert finished == ["result"]


@pytest.mark.parametrize("ffmpeg_path", [None, "ffmpeg-bin"])
def test_voice_message_opus_is_converted_or_sent_as_ogg(monkeypatch, tmp_path, ffmpeg_path):
    source = tmp_path / "voice.opus"
    source.write_bytes(b"OggS-opus")
    converted = tmp_path / "voice-converted.mp3"

    def prepare(exe, src_path, *, transcode_source, max_bytes, min_bitrate_kbps):
        assert transcode_source is bool(ffmpeg_path)
        if not transcode_source:
            return src_path, None
        converted.write_bytes(b"mp3")
        return str(converted), str(converted)

    def request(_method, _url, **kwargs):
        filename, handle, content_type = kwargs["files"]["file"]
        if ffmpeg_path:
            assert (filename, content_type, handle.read()) == ("voice-converted.mp3", "audio/mpeg", b"mp3")
        else:
            assert (filename, content_type, handle.read()) == ("voice.ogg", "audio/ogg", b"OggS-opus")
        return Mock(status_code=200, json=lambda: {"text": "result"})

    monkeypatch.setattr(service.ffmpeg, "prepare_upload", prepare)
    monkeypatch.setattr(service, "request", request)
    worker = make_worker(source, ffmpeg_path=ffmpeg_path)
    finished = []
    worker.finished.connect(finished.append)
    worker.run()
    assert finished == ["result"]
    assert source.read_bytes() == b"OggS-opus"
