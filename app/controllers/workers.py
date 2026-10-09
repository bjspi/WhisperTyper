"""One place that runs request workers on their own QThread and tracks them until they finish."""
from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, Sequence, Tuple

from PyQt6.QtCore import QObject, QThread


class WorkerThreads:
    """Start ``worker.run`` on a fresh QThread; the thread quits on any of the worker's outcomes."""

    def __init__(self) -> None:
        """No threads are running yet."""
        self._running: List[Tuple[QThread, QObject]] = []

    def start(self, worker: Any, outcomes: Sequence[Any], queued: Optional[Callable[[], None]] = None) -> None:
        """Run ``worker`` (whose result handlers are already connected) until an outcome signal fires.

        ``queued`` runs immediately before the thread starts, e.g. to record a timing milestone.
        """
        thread = QThread()
        worker.moveToThread(thread)
        self._running.append((thread, worker))
        thread.started.connect(worker.run)
        for outcome in outcomes:
            outcome.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._forget(thread, worker))
        if queued is not None:
            queued()
        thread.start()

    def _forget(self, thread: QThread, worker: QObject) -> None:
        """Drop the references of a finished worker so both objects can be collected."""
        if (thread, worker) in self._running:
            self._running.remove((thread, worker))
        logging.info("%s thread cleaned up.", type(worker).__name__)

    @property
    def running(self) -> List[QThread]:
        """Threads that have not finished yet."""
        return [thread for thread, _worker in self._running]

    def drain(self) -> None:
        """Ask in-flight threads to finish before exit.

        A QThread that is still running an HTTP request when the interpreter tears down
        triggers Qt's "QThread: Destroyed while thread is still running" abort, so each one
        gets a short grace period to unwind.
        """
        for thread in self.running:
            try:
                if thread.isRunning():
                    thread.quit()
                    thread.wait(2000)
            except Exception as e:
                logging.debug(f"Worker thread drain skipped one thread: {e}")
