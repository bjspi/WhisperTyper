"""Result routing of the transcription pipeline: batch files are collected, everything else is delivered."""
from __future__ import annotations

import copy
import os
from unittest.mock import Mock

import pytest

pytest.importorskip("PyQt6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QObject, pyqtSignal  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

import app.controllers.transcription as pipeline_module  # noqa: E402
from app.audio.store import RecordingStore  # noqa: E402
from app.context import AppContext  # noqa: E402
from app.controllers.transcription import TranscriptionPipeline  # noqa: E402
from app.core.constants import DEFAULT_CONFIG  # noqa: E402

QAPP = QApplication.instance() or QApplication([])


class MemoryConfigStore:
    """ConfigStore stand-in: the test's config, never written to disk."""

    def __init__(self, config):
        """Serve ``config`` as the loaded configuration."""
        self.config = config

    def load(self):
        return self.config, False

    def save(self, _config):
        return True


class FakeWorker(QObject):
    """A transcription worker whose signals the test fires itself."""

    finished = pyqtSignal(str)
    error = pyqtSignal(str, str)
    transcribing = pyqtSignal()
    compressing = pyqtSignal(str)
    diagnostics = pyqtSignal(object)

    def __init__(self, request, timing=None, tr=None):
        """Keep the request and timing like the real worker; ``tr`` is unused."""
        super().__init__()
        self.request = request
        self.timing = timing


def make_pipeline(tmp_path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(replacements_enabled=False, post_rephrasing_entries=[], use_local_px_proxy=False, proxy_url="")
    ctx = AppContext(MemoryConfigStore(config), RecordingStore(str(tmp_path)))
    ctx.notifier = Mock()
    output = Mock()
    output.insert.return_value = True
    return TranscriptionPipeline(ctx, output, Mock(), Mock()), output


def test_a_dictation_finishing_during_a_batch_is_delivered_not_collected(tmp_path, monkeypatch):
    pipeline, output = make_pipeline(tmp_path)
    copied = []
    monkeypatch.setattr(pipeline_module.copykitten, "copy", copied.append)
    starts = []
    monkeypatch.setattr(pipeline, "start", lambda path, output_mode="insert", batch=None, **_kw:
                        starts.append((path, output_mode, batch)))

    pipeline.start_batch(["one.wav", "two.wav"])
    assert starts == [("one.wav", "clipboard", 1)]

    # A hotkey dictation ends while the batch runs: typed as usual, the batch is untouched.
    pipeline.on_transcription_finished("Dictated text.", "insert")
    assert output.insert.call_args.args[0] == "Dictated text."
    pipeline.on_transcription_finished("", "insert")  # No speech: no batch file is skipped either.
    assert pipeline._batch_results == [] and len(starts) == 1 and copied == []

    pipeline.on_transcription_finished("First file.", "clipboard", batch=1)
    assert starts[-1] == ("two.wav", "clipboard", 1) and copied == []
    pipeline.on_transcription_error("boom", "two.wav", "clipboard", batch=1)
    assert copied == ["First file."] and not pipeline._batch_active
    assert pipeline.last_transcription == "First file."

    # A straggler of the finished batch is delivered on its own; the next batch has a new number.
    pipeline.on_transcription_finished("Late.", "clipboard", batch=1)
    assert copied[-1] == "Late."
    pipeline.start_batch(["three.wav"])
    assert starts[-1] == ("three.wav", "clipboard", 2)


def test_start_binds_the_batch_into_the_result_handlers(tmp_path, monkeypatch):
    pipeline, _output = make_pipeline(tmp_path)
    monkeypatch.setattr(pipeline_module, "TranscriptionWorker", FakeWorker)
    monkeypatch.setattr(pipeline_module, "resolve_ffmpeg", lambda _setting: None)
    received = []
    monkeypatch.setattr(pipeline, "on_transcription_finished", lambda text, *args: received.append((text, args[-1])))
    monkeypatch.setattr(pipeline, "on_transcription_error",
                        lambda message, path, *args: received.append((message, args[-1])))

    pipeline.start("one.wav", output_mode="clipboard", batch=3)
    pipeline.start("dictation.wav")
    batch_worker, dictation_worker = [call.args[0] for call in pipeline._workers.start.call_args_list]
    batch_worker.finished.emit("batch text")
    dictation_worker.finished.emit("dictated")
    batch_worker.error.emit("boom", "one.wav")
    assert received == [("batch text", 3), ("dictated", None), ("boom", 3)]
