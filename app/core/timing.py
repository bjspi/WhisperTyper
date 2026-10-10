"""Per-operation, monotonic latency measurements without recording user content.

Milestones are plain log records on the ``whispertyper.timing`` logger. With the queued
logging from :mod:`app.core.log_queue` they cost the producer one enqueue; durations are
computed lazily when the listener formats the record.
"""
from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Callable, Optional

_LOG = logging.getLogger("whispertyper.timing")

# Transport milestones (app.services.http_transport) and the durations derived from them.
_HTTP_DURATIONS = {
    "prepare_ms": ("start", "adapter_start"),
    "tcp_ms": ("tcp_start", "tcp_end"),  # Includes DNS: httpcore resolves inside connect_tcp.
    "proxy_tls_ms": ("proxy_tls_start", "proxy_tls_end"),
    "proxy_tunnel_ms": ("proxy_tunnel_start", "proxy_tunnel_end"),
    "tls_setup_ms": ("tls_start", "tls_end"),
    "connect_ms": ("connect_start", "connect_end"),
    "upload_ms": ("upload_start", "request_sent"),
    "ttfb_ms": ("headers_sent", "first_byte"),
    "request_to_first_byte_ms": ("start", "first_byte"),
    "after_upload_wait_ms": ("request_sent", "first_byte"),
    "response_body_ms": ("headers_received", "body_end"),
    "total_ms": ("start", "end"),
}


class _Durations:
    """Render ``label=ms`` pairs only when the record is formatted, i.e. on the log listener."""

    def __init__(self, events: Mapping[str, int], table: Mapping[str, tuple[str, str]]) -> None:
        """Keep references; the events are complete once the operation/exchange has finished."""
        self._events = events
        self._table = table

    def __str__(self) -> str:
        """Only measured spans appear; absent milestones never become invented durations."""
        events = self._events
        return " ".join(
            f"{label}={(events[end] - events[start]) / 1_000_000:.3f}"
            for label, (start, end) in self._table.items() if start in events and end in events
        )


def queue_http_timing(metadata: dict[str, Any], events: dict[str, int]) -> None:
    """Log one finished HTTP exchange: public routing metadata plus its phase durations."""
    _LOG.info("http_transport %s %s", " ".join(f"{key}={value}" for key, value in metadata.items()),
              _Durations(events, _HTTP_DURATIONS))


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
        "audio_encode_ms": ("audio_encode_start", "audio_encode_end"),
        "audio_compress_ms": ("audio_compress_start", "audio_compress_end"),
        "recording_cleanup_ms": ("file_write_end", "transcription_queued"),
        "worker_setup_ms": ("transcription_queued", "transcription_worker_queued"),
        "worker_queue_ms": ("transcription_worker_queued", "transcription_worker_start"),
        "upload_prepare_ms": ("upload_prepare_start", "upload_prepare_end"),
        "request_setup_ms": ("upload_prepare_end", "transcription_request_start"),
        "transcription_request_ms": ("transcription_request_start", "transcription_response_received"),
        "hedge_trigger_ms": ("transcription_request_start", "transcription_hedge_started"),
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
        _LOG.info("latency_event op=%s source=%s phase=%s at_ms=%.3f elapsed_ms=%.3f",
                  self.operation_id, self.source, phase,
                  (self._epoch_ns + elapsed_ns) / 1_000_000, elapsed_ns / 1_000_000)

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        """Mark ``<name>_start``, and ``<name>_end`` only if the block completes without raising."""
        self.mark(f"{name}_start")
        yield
        self.mark(f"{name}_end")

    def measure_output(self, deliver: Callable[[], bool], outcome: str = "ok") -> None:
        """Time the final delivery and close the operation; a False result or exception is a failure."""
        self.mark("output_start")
        try:
            delivered = deliver()
        except Exception:
            self.finish("output_failed")
            raise
        self.mark("output_end")
        self.finish(outcome if delivered else "output_failed")

    def finish(self, outcome: str) -> None:
        """Log exactly one summary; absent milestones never become invented durations."""
        if self._finished:
            return
        self._finished = True
        now = time.perf_counter_ns()
        self._events["operation_end"] = now
        _LOG.info("latency_summary op=%s source=%s outcome=%s at_ms=%.3f %s",
                  self.operation_id, self.source, outcome,
                  (self._epoch_ns + now - self._started_ns) / 1_000_000,
                  _Durations(self._events, self._DURATIONS))


class _NoTiming(OperationTiming):
    """Null object for callers outside a measured operation: records and logs nothing."""

    def __init__(self) -> None:
        """Start finished, so the inherited mark/finish return without queuing records."""
        self.operation_id = "none"
        self.source = "none"
        self._events = {}
        self._finished = True


#: Shared default for optional timing parameters; avoids ``if timing:`` guards at every milestone.
NO_TIMING: OperationTiming = _NoTiming()
