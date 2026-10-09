"""Per-operation, monotonic latency measurements without recording user content."""
from __future__ import annotations

import atexit
import logging
import queue
import threading
import time
import uuid
from typing import Any, Optional

# SimpleQueue.put never waits for a consumer; all formatting and handler locks stay off the pipeline.
_RECORDS: queue.SimpleQueue[tuple[str, tuple[Any, ...]]] = queue.SimpleQueue()
_START_LOCK = threading.Lock()
_WRITER: Optional[threading.Thread] = None
_SINKS: tuple[logging.Handler, ...] = ()


class _QueuedLogHandler(logging.Handler):
    """Enqueue without a handler lock; the writer alone owns file/console handlers."""

    def __init__(self) -> None:
        """Disable the base handler's lock because SimpleQueue already owns synchronization."""
        super().__init__()
        self.lock = None

    def handle(self, record: logging.LogRecord) -> bool:
        """Apply filters without the base handler's lock, including on Python 3.13+."""
        result = self.filter(record)
        if isinstance(result, logging.LogRecord):
            record = result
        if result:
            self.emit(record)
        return bool(result)

    def emit(self, record: logging.LogRecord) -> None:
        """Dispatch on the writer, otherwise enqueue without formatting the record."""
        if threading.current_thread() is _WRITER:
            _write_log_record(record)
        else:
            _RECORDS.put(("log", (record,)))


def _write_log_record(record: logging.LogRecord) -> None:
    """Keep a failed sink from dropping records for the remaining sinks."""
    for handler in _SINKS:
        try:
            if record.levelno >= handler.level:
                handler.handle(record)
        except Exception:
            continue


def queue_log_handlers() -> None:
    """Move existing root handlers behind the writer during application bootstrap."""
    global _SINKS
    start_timing_logging()
    root = logging.getLogger()
    if any(isinstance(handler, _QueuedLogHandler) for handler in root.handlers):
        return
    _SINKS = tuple(root.handlers)
    root.handlers = [_QueuedLogHandler()]


def log_handlers() -> tuple[logging.Handler, ...]:
    """Return the actual sinks so existing logging settings can adjust their levels."""
    root = logging.getLogger()
    return _SINKS if any(isinstance(handler, _QueuedLogHandler) for handler in root.handlers) else tuple(root.handlers)


def add_log_handler(handler: logging.Handler) -> None:
    """Add a file sink on the writer, or directly if bootstrap did not enable queuing."""
    root = logging.getLogger()
    if any(isinstance(item, _QueuedLogHandler) for item in root.handlers):
        _RECORDS.put(("add_handler", (handler,)))
    else:
        root.addHandler(handler)


def remove_log_handler(handler: logging.Handler) -> None:
    """Remove and close a file sink on the writer so reconfiguration cannot block capture."""
    root = logging.getLogger()
    if any(isinstance(item, _QueuedLogHandler) for item in root.handlers):
        _RECORDS.put(("remove_handler", (handler,)))
    else:
        root.removeHandler(handler)
        handler.close()


def _write_timings() -> None:
    """Drain metadata on one background thread, using the existing logging handlers."""
    global _SINKS
    while True:
        kind, values = _RECORDS.get()
        if kind == "flush":
            values[0].set()
            continue
        try:
            if kind == "add_handler":
                _SINKS = (*_SINKS, values[0])
            elif kind == "remove_handler":
                _SINKS = tuple(handler for handler in _SINKS if handler is not values[0])
                values[0].close()
            elif kind == "log":
                _write_log_record(values[0])
            elif kind == "event":
                logging.info(
                    "latency_event op=%s source=%s phase=%s at_ms=%.3f elapsed_ms=%.3f", *values,
                )
            elif kind == "http":
                metadata, events = values
                pairs = {
                    "prepare_ms": ("start", "adapter_start"),
                    "dns_ms": ("dns_start", "dns_end"),
                    "tcp_ms": ("tcp_start", "tcp_end"),
                    "proxy_tls_ms": ("proxy_tls_start", "proxy_tls_end"),
                    "proxy_tunnel_ms": ("proxy_tunnel_start", "proxy_tunnel_end"),
                    "tls_setup_ms": ("tls_start", "tls_end"),
                    "connect_ms": ("connect_start", "connect_end"),
                    "upload_ms": ("upload_start", "request_sent"),
                    "ttfb_ms": ("headers_sent", "first_byte"),
                    "request_to_first_byte_ms": ("start", "first_byte"),
                    "after_upload_wait_ms": ("request_sent", "first_byte"),
                    "response_headers_ms": ("first_byte", "headers_received"),
                    "response_body_ms": ("headers_received", "body_end"),
                    "total_ms": ("start", "end"),
                }
                details = " ".join(f"{key}={value}" for key, value in metadata.items())
                durations = " ".join(
                    f"{label}={(events[end] - events[start]) / 1_000_000:.3f}"
                    for label, (start, end) in pairs.items() if start in events and end in events
                )
                logging.info("http_transport %s %s", details, durations)
            else:
                operation_id, source, outcome, at_ms, events = values
                durations = " ".join(
                    f"{label}={(events[end] - events[start]) / 1_000_000:.3f}"
                    for label, (start, end) in OperationTiming._DURATIONS.items()
                    if start in events and end in events
                )
                logging.info(
                    "latency_summary op=%s source=%s outcome=%s at_ms=%.3f %s",
                    operation_id, source, outcome, at_ms, durations,
                )
        except Exception:
            # A broken log sink must not kill recording or stop this queue from being drained.
            continue


def start_timing_logging() -> None:
    """Start once during bootstrap, before any hotkey or recording can fire."""
    global _WRITER
    with _START_LOCK:
        if _WRITER is None:
            _WRITER = threading.Thread(target=_write_timings, name="TimingLogWriter", daemon=True)
            _WRITER.start()
            atexit.register(flush_timing_logs)


def flush_timing_logs(timeout: float = 1.0) -> bool:
    """Wait for queued records only at shutdown or in tests, never in the request path."""
    start_timing_logging()
    completed = threading.Event()
    _RECORDS.put(("flush", (completed,)))
    return completed.wait(timeout)


def queue_http_timing(metadata: dict[str, Any], events: dict[str, int]) -> None:
    """Enqueue transport metadata; formatting and sink locks belong to the writer."""
    _RECORDS.put(("http", (metadata, events)))


class OperationTiming:
    """Carry timestamps through sequential capture/worker/Qt handoffs without producer locks."""

    _DURATIONS = {
        "stop_to_api_ms": ("stop", "transcription_response_received"),
        "stop_to_output_ms": ("stop", "output_end"),
        "stop_to_text_commit_ms": ("stop", "text_commit_end"),
        "text_commit_ms": ("text_commit_start", "text_commit_end"),
        "sendinput_dispatch_ms": ("sendinput_dispatch_start", "sendinput_dispatch_end"),
        "clipboard_snapshot_ms": ("clipboard_snapshot_start", "clipboard_snapshot_end"),
        "clipboard_write_ms": ("clipboard_write_start", "clipboard_write_end"),
        "paste_prepare_wait_ms": ("clipboard_write_end", "paste_dispatch_start"),
        "paste_dispatch_ms": ("paste_dispatch_start", "paste_dispatch_end"),
        "paste_settle_wait_ms": ("paste_dispatch_end", "paste_settle_end"),
        "event_queue_ms": ("stop", "stop_handled"),
        "recording_stop_ms": ("stop_handled", "recording_stopped"),
        "recording_tail_wait_ms": ("recording_tail_wait_start", "recording_tail_wait_end"),
        "stop_feedback_ms": ("recording_stopped", "audio_prepare_start"),
        "audio_prepare_ms": ("audio_prepare_start", "audio_prepare_end"),
        "file_write_ms": ("file_write_start", "file_write_end"),
        "recording_cleanup_ms": ("file_write_end", "transcription_queued"),
        "worker_setup_ms": ("transcription_queued", "transcription_worker_queued"),
        "worker_queue_ms": ("transcription_worker_queued", "transcription_worker_start"),
        "upload_prepare_ms": ("upload_prepare_start", "upload_prepare_end"),
        "request_setup_ms": ("upload_prepare_end", "transcription_request_start"),
        "transcription_request_ms": ("transcription_request_start", "transcription_response_received"),
        "stop_to_request_sent_ms": ("stop", "transcription_http_request_sent"),
        "stop_to_first_byte_ms": ("stop", "transcription_http_first_byte"),
        "response_parse_ms": ("transcription_response_received", "transcription_response_parsed"),
        "result_queue_ms": ("transcription_worker_ready", "transcription_result_received"),
        "result_processing_ms": ("transcription_result_received", "result_processed"),
        "replacements_ms": ("replacements_start", "replacements_end"),
        "rephrase_setup_ms": ("rephrase_queued", "rephrase_worker_queued"),
        "rephrase_queue_ms": ("rephrase_worker_queued", "rephrase_worker_start"),
        "rephrase_request_ms": ("rephrase_request_start", "rephrase_response_received"),
        "rephrase_parse_ms": ("rephrase_response_received", "rephrase_response_parsed"),
        "rephrase_result_queue_ms": ("rephrase_worker_ready", "output_start"),
        "output_ms": ("output_start", "output_end"),
        "total_ms": ("operation_start", "operation_end"),
    }

    def __init__(self, source: str = "file", detected_ns: Optional[int] = None) -> None:
        """Start at listener detection when supplied, otherwise at operation creation."""
        self.operation_id = uuid.uuid4().hex[:12]
        self.source = source
        self._started_ns = detected_ns if detected_ns is not None else time.perf_counter_ns()
        self._epoch_ns = time.time_ns() - (time.perf_counter_ns() - self._started_ns)
        self._events: dict[str, int] = {"operation_start": self._started_ns}
        self._finished = False
        if source == "recording":
            self.mark("stop", self._started_ns)

    def mark(self, phase: str, at_ns: Optional[int] = None) -> None:
        """Record a milestone once and log its offset from the operation's start."""
        now = at_ns if at_ns is not None else time.perf_counter_ns()
        if self._finished or phase in self._events:
            return
        self._events[phase] = now
        elapsed_ns = now - self._started_ns
        _RECORDS.put(("event", (
            self.operation_id, self.source, phase,
            (self._epoch_ns + elapsed_ns) / 1_000_000, elapsed_ns / 1_000_000,
        )))

    def finish(self, outcome: str) -> None:
        """Log exactly one summary; absent milestones never become invented durations."""
        if self._finished:
            return
        self._finished = True
        now = time.perf_counter_ns()
        self._events["operation_end"] = now
        _RECORDS.put(("summary", (
            self.operation_id, self.source, outcome,
            (self._epoch_ns + now - self._started_ns) / 1_000_000, self._events,
        )))
