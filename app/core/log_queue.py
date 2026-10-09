"""Application-wide asynchronous logging on the stdlib QueueHandler/QueueListener.

Producers (hotkeys, audio capture, workers) only enqueue unformatted records; formatting and
sink I/O run on one listener thread, so a slow disk or a stalled handler never delays the
recording pipeline. Sinks are kept in a copy-on-write tuple, so adding/removing them never
waits for the listener either.
"""
from __future__ import annotations

import atexit
import logging
import queue
import sys
import threading
from logging.handlers import QueueHandler, QueueListener
from typing import Any, Callable, Optional

_QUEUE: queue.SimpleQueue[logging.LogRecord] = queue.SimpleQueue()
_SINKS: tuple[logging.Handler, ...] = ()
_LISTENER: Optional[QueueListener] = None
_INSTALL_LOCK = threading.Lock()


class _InProcessQueueHandler(QueueHandler):
    """Enqueue the record as-is: formatting happens on the listener, not on the producer."""

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        """Skip QueueHandler's eager formatting; nothing crosses a process boundary."""
        return record


def _control(action: Callable[[], None]) -> logging.LogRecord:
    """A queue-only record that runs ``action`` on the listener thread, in queue order."""
    record = logging.makeLogRecord({"msg": "log_queue control"})
    record.log_queue_action = action
    return record


class _SinkDispatcher(logging.Handler):
    """Listener-side fan-out that isolates sinks: one failing sink never stops the others."""

    def handle(self, record: logging.LogRecord) -> bool:
        """Run control actions, otherwise hand the record to every sink at or below its level."""
        action = getattr(record, "log_queue_action", None)
        if action is not None:
            action()
            return True
        for sink in _SINKS:
            try:
                if record.levelno >= sink.level:
                    sink.handle(record)
            except Exception:
                continue
        return True

    def emit(self, record: logging.LogRecord) -> None:
        """Unused: ``handle`` dispatches directly."""


def install() -> None:
    """Move the root handlers behind the queue (idempotent; called once at bootstrap)."""
    global _SINKS, _LISTENER
    with _INSTALL_LOCK:
        if _LISTENER is not None:
            return
        root = logging.getLogger()
        _SINKS = tuple(root.handlers)
        root.handlers = [_InProcessQueueHandler(_QUEUE)]
        _LISTENER = QueueListener(_QUEUE, _SinkDispatcher())
        _LISTENER.start()
        atexit.register(shutdown)
        # Qt aborts after an uncaught slot exception, skipping atexit: flush from the hooks too.
        sys.excepthook = _flushing(sys.excepthook)
        threading.excepthook = _flushing(threading.excepthook)


def _flushing(hook: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap an exception hook so queued records reach the sinks before a possible abort."""
    def wrapper(*args: Any) -> Any:
        try:
            return hook(*args)
        finally:
            flush_on_crash()
    return wrapper


def is_installed() -> bool:
    """Whether records currently flow through the listener thread."""
    return _LISTENER is not None


def sinks() -> tuple[logging.Handler, ...]:
    """The real output handlers, so settings can adjust their levels."""
    return _SINKS if is_installed() else tuple(logging.getLogger().handlers)


def add_sink(handler: logging.Handler) -> None:
    """Attach an output handler without waiting for the listener."""
    global _SINKS
    if not is_installed():
        logging.getLogger().addHandler(handler)
        return
    _SINKS = (*_SINKS, handler)


def remove_sink(handler: logging.Handler) -> None:
    """Detach a handler now and close it on the listener, after its queued records."""
    global _SINKS
    if not is_installed():
        logging.getLogger().removeHandler(handler)
        handler.close()
        return
    _SINKS = tuple(sink for sink in _SINKS if sink is not handler)
    _QUEUE.put(_control(handler.close))


def flush(timeout: float = 1.0) -> bool:
    """Wait until every record queued so far has reached the sinks (shutdown, crashes, tests)."""
    if not is_installed():
        return True
    done = threading.Event()
    _QUEUE.put(_control(done.set))
    return done.wait(timeout)


def shutdown() -> None:
    """Drain the queue and stop the listener at interpreter exit."""
    global _LISTENER
    with _INSTALL_LOCK:
        listener, _LISTENER = _LISTENER, None
    if listener is not None:
        listener.stop()
        root = logging.getLogger()
        root.handlers = list(_SINKS)


def flush_on_crash(*_args: Any) -> None:
    """Best-effort flush from an excepthook, so the traceback's context reaches the log file."""
    try:
        flush(2.0)
    except Exception:
        pass
