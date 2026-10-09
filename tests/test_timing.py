"""Latency accuracy and logging backpressure regression tests (no Qt or audio hardware)."""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import pytest

from app.core import log_queue, timing


def test_recording_summary_uses_detection_time_and_monotonic_durations(monkeypatch, caplog):
    assert log_queue.flush()
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
        assert log_queue.flush()
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
        assert log_queue.flush()
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
        assert log_queue.flush()
    assert first.operation_id != second.operation_id
    first_summary = next(line for line in caplog.messages if line.startswith(f"latency_summary op={first.operation_id}"))
    second_summary = next(line for line in caplog.messages if line.startswith(f"latency_summary op={second.operation_id}"))
    assert "rephrase_request_ms=" in first_summary
    assert "transcription_request_ms=" not in first_summary
    assert "rephrase_request_ms=" not in second_summary


def test_queue_handler_defers_formatting_to_the_listener():
    handler = log_queue._InProcessQueueHandler(log_queue._QUEUE)
    record = logging.makeLogRecord({"msg": "value=%s", "args": ("lazy",)})
    assert handler.prepare(record) is record
    assert record.msg == "value=%s" and record.args == ("lazy",)


@pytest.mark.parametrize("stall", ["handler_lock", "slow_emit"])
def test_blocked_sink_does_not_block_producers_or_handler_reconfiguration(stall):
    # A separate process keeps the application-wide logging setup away from pytest's handlers.
    script = r'''
import logging
import sys
import threading
from app.core import log_queue, timing

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
log_queue.install()
log_queue.install()  # Repeated bootstrap must not wrap the queue again.
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
    log_queue.remove_sink(sink)
    log_queue.add_sink(BrokenSink())
    log_queue.add_sink(sink)
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
assert log_queue.flush(3)
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
