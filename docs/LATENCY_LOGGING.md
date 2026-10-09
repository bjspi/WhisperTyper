# Measuring transcription latency

Enable **file logging** in General settings, then open `WhisperTyper.log` from the
menu. Timing records are logged at INFO, so debug logging is optional. Timing
records contain operation IDs, phases and durations, never transcripts, prompts,
API keys or file paths. Existing log redaction still applies to other messages.

Each recording, file transcription, retry or standalone rephrasing request gets
its own `op` ID. Filter by that ID to follow a request through the workers and text
delivery, including any rephrasing fallback. `latency_event` records each milestone;
`latency_summary` records the outcome and durations in milliseconds.

For a recording, the most useful measurements are:

| Field | Interval |
| --- | --- |
| `stop_to_api_ms` | Stop detection → complete transcription HTTP response body |
| `stop_to_output_ms` | Stop detection → final text delivery routine completed |
| `event_queue_ms` | Hotkey listener detection → stop handler on the GUI thread |
| `recording_stop_ms` | Stop handler → recorder stopped / capture thread joined (includes stop UI work) |
| `stop_feedback_ms` | Recorder stopped → audio processing started (includes stop sound dispatch) |
| `audio_prepare_ms` | Read/collect PCM, validate, resample and apply gain |
| `file_write_ms` | Write the recording file |
| `recording_cleanup_ms` | Update recording actions and prune older recordings before worker setup |
| `worker_setup_ms` / `worker_queue_ms` | Configure the worker / wait for its thread to run |
| `upload_prepare_ms` | Prepare the file for upload |
| `request_setup_ms` | Prepared upload → begin HTTP request |
| `transcription_request_ms` | Begin HTTP request → complete response body |
| `response_parse_ms` | Complete response → extract the transcription from JSON |
| `result_queue_ms` / `result_processing_ms` | Worker ready → GUI callback / route the result |
| `rephrase_setup_ms` / `rephrase_queue_ms` | Configure rephrasing / wait for its thread |
| `rephrase_request_ms` / `rephrase_parse_ms` | Rephrasing HTTP request / parse its result |
| `rephrase_result_queue_ms` | Rephrasing worker ready → text delivery starts |
| `output_ms` | Copy or paste routine, including its UI feedback |
| `total_ms` | Operation start → completion or failure |

Times use a monotonic clock. `elapsed_ms` is relative to the operation start;
`at_ms` is the captured time mapped to Unix epoch milliseconds. The normal log-line
date reflects background writing time and can be later. Use the captured values
for analysis, rather than subtracting log-line dates.

Hotkey measurements start when the listener receives the event; they cannot
measure the delay before the operating system delivers it. Button/tray stops start
when their handler runs. File uploads and retries have no stop event and therefore
no `stop_to_*` values. A missing phase produces no invented duration. A batch file
ends with `batch_buffered`; the combined clipboard write is outside that per-file
operation. Text delivery completion means the app dispatched the paste or copied
the text, not that another application acknowledged it.

Currently the HTTP request duration combines multipart preparation, connection,
upload, server processing and response download. It is **not TTFB**. DNS, TCP,
TLS, connection reuse, upload completion and TTFB are separate transport work.

## Logging does not wait for the disk

The writer starts during bootstrap, before recording/hotkeys. The timing path
captures timestamps and submits metadata to `SimpleQueue`; it never waits for a
log consumer or takes a file/console handler lock. Regular application logging is
queued too, so a handler busy with a timing record cannot stall the next ordinary
log message. Formatting, handler locks, rotation and file writes run on the daemon
writer. Removing/closing a file handler is queued as well. No new synchronous
flush is performed during recording, upload or result delivery.

Capturing timestamps and queueing still have a small CPU/allocation cost; this
does not promise zero overhead. The queue is unbounded so producers do not wait:
a persistently blocked sink can grow memory usage. Shutdown allows one second to
drain queued entries; a blocked sink or forced process termination can lose pending
logs. Logging failures do not stop the transcription pipeline.
