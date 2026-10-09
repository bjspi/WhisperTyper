"""Latency accuracy and logging backpressure regression tests (no Qt or audio hardware)."""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import pytest

from app.core import timing


def test_recording_summary_uses_detection_time_and_monotonic_durations(monkeypatch, caplog):
    assert timing.flush_timing_logs()
    clock = iter([12_000_000, 15_000_000])
    monkeypatch.setattr(timing.time, "perf_counter_ns", lambda: next(clock))
    monkeypatch.setattr(timing.time, "time_ns", lambda: 1_700_000_000_000_000_000)
    with caplog.at_level(logging.INFO):
        operation = timing.OperationTiming("recording", 10_000_000)
        operation.mark("stop_handled", 11_000_000)
        operation.mark("transcription_request_start", 12_000_000)
        operation.mark("transcription_response_received", 13_000_000)
        operation.mark("output_start", 14_000_000)
        operation.mark("output_end", 15_000_000)
        operation.finish("ok")
        assert timing.flush_timing_logs()
    summary = next(line for line in caplog.messages if line.startswith("latency_summary"))
    assert "stop_to_api_ms=3.000" in summary
    assert "stop_to_output_ms=5.000" in summary
    assert "event_queue_ms=1.000" in summary
    assert "transcription_request_ms=1.000" in summary
    assert "output_ms=1.000" in summary
    assert "total_ms=5.000" in summary
    assert "at_ms=1700000000003.000" in summary
    assert "ttfb" not in summary  # A complete requests.post response is not a first byte.


def test_failed_file_has_no_invented_stop_or_output_duration(caplog):
    with caplog.at_level(logging.INFO):
        operation = timing.OperationTiming("file")
        operation.mark("transcription_request_start")
        operation.mark("transcription_failed")
        operation.finish("transcription_failed")
        operation.finish("ok")
        operation.mark("output_end")
        assert timing.flush_timing_logs()
    summaries = [line for line in caplog.messages if line.startswith("latency_summary")]
    assert len(summaries) == 1
    assert "outcome=transcription_failed" in summaries[0]
    assert "stop_to_" not in summaries[0]
    assert "transcription_request_ms" not in summaries[0]
    assert "output_end" not in caplog.text


def test_interleaved_jobs_have_independent_milestones(caplog):
    with caplog.at_level(logging.INFO):
        first = timing.OperationTiming("recording")
        second = timing.OperationTiming("file")
        first.mark("rephrase_request_start")
        second.mark("transcription_request_start")
        first.mark("rephrase_response_received")
        second.finish("transcription_failed")
        first.finish("rephrase_failed_fallback")
        assert timing.flush_timing_logs()
    assert first.operation_id != second.operation_id
    first_summary = next(line for line in caplog.messages if line.startswith(f"latency_summary op={first.operation_id}"))
    second_summary = next(line for line in caplog.messages if line.startswith(f"latency_summary op={second.operation_id}"))
    assert "rephrase_request_ms=" in first_summary
    assert "transcription_request_ms=" not in first_summary
    assert "rephrase_request_ms=" not in second_summary


def test_queue_handler_applies_filters_without_using_base_handler_lock(monkeypatch):
    handler = timing._QueuedLogHandler()
    records = []
    monkeypatch.setattr(handler, "emit", records.append)
    record = logging.makeLogRecord({"msg": "original"})
    handler.addFilter(lambda _record: False)
    assert not handler.handle(record)
    assert records == []
    handler.filters.clear()
    assert handler.lock is None
    assert handler.handle(record)
    assert records == [record]
    if sys.version_info >= (3, 12):
        replacement = logging.makeLogRecord({"msg": "filtered"})
        handler.addFilter(lambda _record: replacement)
        assert handler.handle(record)
        assert records[-1] is replacement


@pytest.mark.parametrize("stall", ["handler_lock", "slow_emit"])
def test_blocked_sink_does_not_block_producers_or_handler_reconfiguration(stall):
    # A separate process keeps the application-wide logging setup away from pytest's handlers.
    script = r'''
import logging
import sys
import threading
from app.core import timing

entered = threading.Event()
release = threading.Event()
produced = threading.Event()
records = []

class Sink(logging.Handler):
    def handle(self, record):
        entered.set()
        return super().handle(record)

    def emit(self, record):
        if sys.argv[1] == "slow_emit":
            assert release.wait(5)
        records.append(record.getMessage())

class BrokenSink(logging.Handler):
    def emit(self, record):
        raise OSError("synthetic disk error")

sink = Sink()
logging.getLogger().handlers = [sink]
logging.getLogger().setLevel(logging.INFO)
timing.queue_log_handlers()
timing.queue_log_handlers()  # Repeated bootstrap must not wrap the queue again.
if sys.argv[1] == "handler_lock":
    sink.acquire()
logging.info("block sink")
assert entered.wait(2)

def produce():
    operation = timing.OperationTiming("recording")
    for _ in range(100):
        operation.mark("stop_handled")
        logging.info("ordinary queued log %s", "sensitive-content")
    operation.finish("ok")
    timing.queue_http_timing({"op": operation.operation_id, "stage": "transcription"},
                            {"start": 1, "headers_sent": 2, "first_byte": 3, "end": 4})
    timing.remove_log_handler(sink)
    timing.add_log_handler(BrokenSink())
    timing.add_log_handler(sink)
    logging.info("after reconfiguration")
    produced.set()

producer = threading.Thread(target=produce, daemon=True)
producer.start()
try:
    assert produced.wait(1), "producer waited for a blocked log sink"
finally:
    release.set()
    if sys.argv[1] == "handler_lock":
        sink.release()
producer.join(2)
assert timing.flush_timing_logs(3)
assert "after reconfiguration" in records  # A broken sink must not kill the writer.
summaries = [line for line in records if line.startswith("latency_summary")]
assert len(summaries) == 1
assert any(line.startswith("http_transport") for line in records)
assert "sensitive-content" not in summaries[0]
assert sum("phase=stop_handled " in line for line in records) == 1
'''
    result = subprocess.run(
        [sys.executable, "-c", script, stall], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
