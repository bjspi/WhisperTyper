"""Recording → transcription → output through the real controllers, with a fake microphone and API."""
from __future__ import annotations

import copy
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock

import pytest

pytest.importorskip("PyQt6.QtWidgets")
pytest.importorskip("pyaudio")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

import app.controllers.recording as recording_module  # noqa: E402
import app.controllers.text_output as text_output_module  # noqa: E402
from app.audio.capture import HotMicCapture  # noqa: E402
from app.audio.store import RecordingStore  # noqa: E402
from app.context import AppContext  # noqa: E402
from app.controllers.permissions import MacPermissions  # noqa: E402
from app.controllers.recording import RecordingController  # noqa: E402
from app.controllers.text_output import TextOutput  # noqa: E402
from app.controllers.transcription import TranscriptionPipeline  # noqa: E402
from app.controllers.warmup import WarmupScheduler  # noqa: E402
from app.controllers.workers import WorkerThreads  # noqa: E402
from app.core.constants import DEFAULT_CONFIG  # noqa: E402
from app.core.timing import OperationTiming  # noqa: E402

TONE = b"\x00\x10\x00\xf0" * 512  # 1,024 non-silent 16-bit samples
# One application for the whole module: widgets such as the tooltip singleton outlive a test.
QAPP = QApplication.instance() or QApplication([])


class FakeApi(BaseHTTPRequestHandler):
    """Transcribes every upload as a phrase with a typo; chat completions answer REPHRASED."""

    uploads: list[int] = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path.endswith("/audio/transcriptions"):
            FakeApi.uploads.append(len(body))
            payload = {"text": "hello wrold"}
        else:
            payload = {"choices": [{"message": {"content": "REPHRASED"}}]}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_args):
        pass


class FakeStream:
    """Delivers a tone in real-time-sized blocks; nothing extra is buffered at stop."""

    def read(self, frames, **_kwargs):
        time.sleep(0.004)
        return (TONE * (frames // 1024 + 1))[: frames * 2]

    def get_read_available(self):
        return 0


class FakeMicrophone:
    """The MicrophoneInput surface HotMicCapture uses, without PyAudio."""

    target_samplerate = samplerate = 16000
    chunk_size = 1024

    def open(self):
        return FakeStream()

    def close(self):
        pass


class MemoryConfigStore:
    """ConfigStore stand-in: the test's config, never written to disk."""

    def __init__(self, config):
        """Serve ``config`` as the loaded configuration."""
        self.config = config

    def load(self):
        return self.config, False

    def save(self, _config):
        pass


class Paster(TextOutput):
    """Records simulated paste keys instead of pressing them."""

    keys: list[str]

    def _simulate_key_combination(self, char: str) -> bool:
        assert threading.current_thread() is threading.main_thread()  # Output runs on the GUI thread.
        self.keys.append(char)
        return True


@pytest.fixture
def api():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeApi)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


@pytest.mark.parametrize("rephrase", [False, True])
def test_hotkey_stop_reaches_output_with_ordered_milestones(api, tmp_path, monkeypatch, rephrase):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(
        api_endpoint=f"{api}/v1/audio/transcriptions", rephrasing_api_url=f"{api}/v1/chat/completions",
        api_key_profiles=[{"id": "p", "name": "Local", "provider": "custom", "key": "test-key"}],
        transcription_key_profile_id="p", rephrasing_key_profile_id="p", rephrasing_model="test-model",
        min_recording_seconds=0, restore_clipboard=False, use_local_px_proxy=False, proxy_url="",
        rephrase_use_selection_context=False, liveprompt_enabled=False, generic_rephrase_enabled=rephrase,
        replacements_enabled=True, replacements_rules="wrold ; world", post_rephrasing_entries=[],
        windows_keep_mic_hot=True,
    )
    monkeypatch.setattr(recording_module, "is_WINDOWS", True)  # Exercise the warm-microphone path everywhere.
    pasted: list[str] = []
    monkeypatch.setattr(text_output_module.copykitten, "copy", pasted.append)
    operations: list[OperationTiming] = []

    class RecordedTiming(OperationTiming):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            operations.append(self)

    monkeypatch.setattr(recording_module, "OperationTiming", RecordedTiming)

    ctx = AppContext(MemoryConfigStore(config), RecordingStore(str(tmp_path)))
    workers = WorkerThreads()
    warmup = WarmupScheduler(config, is_recording=lambda: False, rephrasing_first=lambda: False)
    warmup.http.schedule = Mock()
    permissions = MacPermissions(ctx, dialog_parent=lambda: None)
    output = Paster(config, ctx.tr, ctx.notifier.show, permissions.warn)
    output.keys = []
    pipeline = TranscriptionPipeline(ctx, output, warmup, workers)
    recording = RecordingController(ctx, permissions=permissions, warmup=warmup, output=output, pipeline=pipeline,
                                    sounds=Mock(), open_settings=Mock(), palette_anchor=lambda: None)
    recording.hot_mic = HotMicCapture(FakeMicrophone(), idle_expired=lambda: False,
                                      on_block=recording._update_latest_audio_level)
    states: list[bool] = []
    recording.state_changed.connect(states.append)

    try:
        recording.toggle()
        assert recording.is_recording and states == [True]
        warmup.http.schedule.assert_called()  # Prewarm on recording start.
        time.sleep(0.15)
        detected_ns = time.perf_counter_ns()
        recording.toggle(detected_ns=detected_ns)
        deadline = time.monotonic() + 10
        while not output.keys and time.monotonic() < deadline:
            QAPP.processEvents()
            time.sleep(0.01)
        for _ in range(20):
            QAPP.processEvents()
    finally:
        recording.hot_mic.stop()
        workers.drain()
        warmup.close()

    expected_text = "REPHRASED" if rephrase else "hello world"
    assert output.keys == ["v"] and pasted == [expected_text]
    assert pipeline.last_transcription == expected_text
    assert states == [True, False] and FakeApi.uploads
    assert len(operations) == 1
    events = list(operations[0]._events)
    assert operations[0]._events["stop"] == detected_ns
    milestones = [
        "stop", "stop_handled", "recording_stopped", "audio_prepare_start", "audio_prepare_end",
        "file_write_start", "file_write_end", "transcription_queued", "transcription_worker_queued",
        "transcription_worker_start", "upload_prepare_start", "upload_prepare_end", "transcription_request_start",
        "transcription_http_request_sent", "transcription_http_first_byte", "transcription_response_received",
        "transcription_response_parsed", "transcription_worker_ready", "transcription_result_received",
        "replacements_start", "replacements_end", "result_processed",
    ]
    if rephrase:
        milestones += ["rephrase_queued", "rephrase_worker_queued", "rephrase_worker_start", "rephrase_request_start",
                       "rephrase_response_received", "rephrase_response_parsed", "rephrase_worker_ready"]
    milestones += ["output_start", "text_commit_start", "clipboard_write_start", "clipboard_write_end",
                   "paste_dispatch_start", "paste_dispatch_end", "text_commit_end", "paste_settle_end",
                   "output_end", "operation_end"]
    positions = [events.index(name) for name in milestones]
    assert positions == sorted(positions), [name for name in events if name in milestones]
    assert "recording_tail_timeout" not in events
