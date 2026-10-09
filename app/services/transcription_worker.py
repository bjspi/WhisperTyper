"""Runs one transcription request off the GUI thread."""
from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal

from app.core.timing import OperationTiming
from app.services.transcription import TranscriptionError, TranscriptionRequest, Translate, transcribe


class TranscriptionWorker(QObject):
    """Qt wrapper that runs :func:`app.services.transcription.transcribe` on a worker thread."""

    finished = pyqtSignal(str)
    error = pyqtSignal(str, str)
    # Emitted once pre-processing (video extraction / compression) finishes and the upload begins,
    # so the tray spinner can switch to the "transcribing…" phase.
    transcribing = pyqtSignal()
    # Emitted just before an oversized file is compressed, so the tray can show a distinct
    # "compressing…" spinner during the (blocking) re-encode before the transcription phase.
    compressing = pyqtSignal(str)

    def __init__(self, request: TranscriptionRequest, timing: Optional[OperationTiming] = None,
                 tr: Optional[Translate] = None) -> None:
        """Keep the request snapshot, the operation timing and the translation lookup.

        Args:
            request: Everything the request needs, snapshotted on the GUI thread.
            timing: This operation's timing state, shared with the GUI result callbacks.
            tr: Translation lookup for user-facing errors (a plain dict read, safe off the GUI thread).
        """
        super().__init__()
        self.request = request
        self.audio_path = request.audio_path
        self.timing = timing or OperationTiming()
        self.tr: Translate = tr or (lambda key, **_kwargs: key)

    def run(self) -> None:
        """Execute the transcription and emit ``finished`` or ``error``."""
        logging.info("TranscriptionWorker started.")
        self.timing.mark("transcription_worker_start")
        try:
            text = transcribe(self.request, self.timing, self.tr,
                              on_compressing=self.compressing.emit, on_transcribing=self.transcribing.emit)
        except TranscriptionError as e:
            self._fail(str(e))
            return
        except Exception as e:
            self._fail(self.tr("transcription_worker_error", error=e))
            return
        self.timing.mark("transcription_worker_ready")
        self.finished.emit(text)

    def _fail(self, message: str) -> None:
        """Report a failure; the source file is kept so the user can retry."""
        logging.error(message)
        self.timing.mark("transcription_failed")
        self.error.emit(message, self.audio_path)
