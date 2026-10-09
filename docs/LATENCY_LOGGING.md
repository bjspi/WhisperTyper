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
| `stop_to_request_sent_ms` | Stop detection → request headers/body handed to the socket |
| `stop_to_first_byte_ms` | Stop detection → first HTTP response byte observed |
| `stop_to_output_ms` | Stop detection → final text delivery routine completed |
| `stop_to_text_commit_ms` | Stop detection → input dispatch completed (paste or direct Unicode input) |
| `text_commit_ms` | Start insertion routine → input dispatch completed |
| `sendinput_dispatch_ms` | Direct Unicode input attempt, including event preparation and native dispatch |
| `clipboard_snapshot_ms` / `clipboard_write_ms` | Preserve original clipboard / write the result to it |
| `paste_prepare_wait_ms` | Clipboard write completed → paste dispatch starts (100 ms wait unless fast paste is enabled) |
| `paste_dispatch_ms` | Ctrl/Cmd+V dispatch, including macOS System Events/fallback if used |
| `paste_settle_wait_ms` | Paste dispatch completed → settling phase ends (100 ms wait unless fast paste is enabled) |
| `event_queue_ms` | Hotkey listener detection → stop handler on the GUI thread |
| `recording_stop_ms` | Stop handler → recorder stopped / capture thread joined (includes stop UI work) |
| `recording_tail_wait_ms` | Windows keep-mic-hot: wait for an already active read and buffered samples to be retained |
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
the text, not that another application acknowledged it. `text_commit_end` is
captured immediately after successful input dispatch, before any existing settling
wait, clipboard-restore scheduling and final UI bookkeeping. It does **not** prove
when a foreign application's field changed or was rendered. No UI polling or extra
wait was added for measurement. Clipboard-only output has `clipboard_write_ms`
but no invented paste/text-commit milestones. A failed clipboard write or paste
dispatch records `text_commit_failed`, omits `text_commit_end`, and finishes with
`outcome=output_failed`. Delayed clipboard restoration remains outside text-commit
timing: 500 ms after the normal routine, or 600 ms after dispatch in fast-paste mode.

## Recording boundary on Windows

With **Keep mic hot**, stopping a recording now attaches the active input read to
that recording before disabling capture. The capture thread retains the in-flight
block and reads a snapshot of already available driver-buffered samples before the
WAV is written. There is no fixed post-recording sleep and no wait if a read is not
active. At 16 kHz, the normal 1,024-sample block spans 64 ms; previously a stop during
that read could discard the entire last block. Finishing that read can naturally
take the remaining fraction of a block and include a little audio after the keypress.

`recording_tail` logs the operation ID, whether the tail was retained, sample counts
for the pending/buffered data and the sample rate, without audio contents. Timing
events distinguish the read completing, tail being saved, drain/read failures and
the actual wait. A stalled driver has a 250 ms wait ceiling: `recording_tail_timeout`
is explicit, and late data cannot change a WAV already submitted for transcription.
Cancellation continues to discard pending audio. The per-recording PyAudio reader
already retains its active read and is joined on stop; the native macOS recorder
continues to use AVAudioRecorder's own stop/finalization. This Windows race fix does
not guarantee that every provider will transcribe every word in intact audio.

The existing `transcription_request_ms` includes the entire request and is **not
TTFB**. Each wire exchange now also writes an `http_transport` summary, correlated
by `op`, with `stage=transcription`, `rephrase`, `connection_test` or `prewarm`.
Redirects have separate numbered `exchange` records. HTTP errors and network
failures retain the milestones actually reached, without invented durations.

| Transport field | Interval / meaning |
| --- | --- |
| `prepare_ms` | API call start → adapter, including multipart/JSON assembly and environment settings |
| `dns_ms` | Resolve the target or proxy hostname |
| `tcp_ms` | TCP setup, including fallback between resolved addresses |
| `proxy_tls_ms` / `proxy_tunnel_ms` | TLS to an HTTPS proxy / CONNECT tunnel setup |
| `tls_setup_ms` | Origin TLS setup: context/CA preparation, handshake and certificate verification |
| `connect_ms` | Entire connection setup including DNS/TCP/TLS and any proxy tunnel |
| `upload_ms` | First body send → complete request sent; multipart framing is included |
| `ttfb_ms` | Request headers sent → first response byte, including upload time |
| `request_to_first_byte_ms` | API call start → first response byte, including preparation and connection setup |
| `after_upload_wait_ms` | Complete request sent → first response byte |
| `response_headers_ms` | First byte → complete final response headers |
| `response_body_ms` | Complete headers → complete consumed/decompressed response body |
| `total_ms` | API call start → complete response or failure |
| `connection` / `reused` | Socket identity / whether this exchange reused an existing socket |
| `error` / `last_phase` | Exception class (without private exception text) / last milestone reached before completion or failure |
| `upload_bytes` / `response_body_bytes` | Sent body bytes / consumed response bytes after decompression |

Transport milestones also appear as `latency_event` phases such as
`transcription_http_request_sent` and `transcription_http_first_byte` for operations
with timing state. Use their captured `at_ms` to determine when the POST finished
sending, rather than the log writer's line date. New connections have DNS/TCP/TLS
fields; reused ones omit them. `tls_setup_ms` intentionally includes certificate
verification, rather than pretending to isolate only the cryptographic handshake.
TTFB observes the first socket read before status/header parsing (an informational
HTTP response, if present, counts as the first byte). Sending completion means the
OS accepted the bytes, not that the server acknowledged reading all of them.
`after_upload_wait_ms` combines transit and provider processing; the client cannot
split the provider's queue from inference without server-side timing data.

## Persistent connections and background warming

All transcription, rephrasing and API connection-test calls share bounded urllib3
HTTP/1.1 pools, even when each call uses a new Qt worker thread or a different
Groq key. Headers and cookies remain isolated per logical request. TLS verification,
environment CA bundles and existing explicit/system/px proxy routing remain active.
TCP_NODELAY and OS TCP keepalive are enabled; failed paid POSTs are not retried.
Busy pools create another socket instead of waiting for a warm-up request.

A daemon performs auth-free HEAD requests at startup, after saving settings and
when recording begins. While recording or for five minutes after recent activity,
a timer refreshes each configured origin at most every 20 seconds. Groq/OpenAI
use their `/models` path; custom endpoints use the origin root. URL credentials,
queries, API keys, netrc origin credentials and audio are excluded. Redirects are
disabled. A 401/404/405 still proves that the connection reached the server.
Warm-up has short timeouts and never delays capture, upload or shutdown waiting
for completion. The server can still close idle connections; urllib3 reconnects
when required. There is no guarantee of reuse during overlapping requests.

HTTP/HTTPS proxies receive full socket measurements. Optional SOCKS proxy support
keeps requests' own connection implementation and reports overall/body timing,
without inventing DNS/TCP/TLS/TTFB measurements for that implementation.

## Optional Windows text input

General → Insertion options offers an experimental Windows-only direct Unicode `SendInput`
mode, disabled by default. It uses ctypes and one batch of UTF-16 key-down/key-up
events, including surrogate pairs for emoji, without accessing the clipboard,
sleeping or scheduling a clipboard restore. Both transcription and rephrasing
insertion use it; deliberate clipboard-only output is unchanged. Newlines become
Unicode CR characters; no physical Return/Tab shortcuts are synthesized. Target
applications may ignore Unicode packets, line breaks or tabs, so this mode needs
testing in the user's actual editors. Held modifier keys reject the direct attempt.

`text_input` records correlate by `op` and identify `mode=sendinput` or `clipboard`,
accepted/total event counts and automatic fallback, without text contents.
`sendinput_dispatch_ms` measures the direct attempt; `text_commit_ms` and
`stop_to_text_commit_ms` still end at dispatch completion, not target acknowledgement.
Successful direct input has no clipboard or paste timings. Fallback operations show
both the direct attempt and the existing clipboard timings.

The separate fallback checkbox defaults on and retries via the existing paste path
only when no events were accepted (or input was rejected before injection). Partial
acceptance and unknown failures stop with a notification rather than risking duplicate
text. A target ignoring fully accepted events cannot be detected automatically; the
user must disable direct input for that application. UIPI still prevents injection into
applications running with higher privileges.

The independent **Fast Copy/Paste** checkbox is available on Windows and macOS,
and disabled by default. On macOS, Cmd+V still uses `osascript` first; this option
does not eliminate subprocess/System Events latency. Real Mac testing is pending.
It skips the 100 ms waits before and after paste and disables PyAutoGUI's additional
post-paste pause for this call only (its failsafe and copy-selection waits remain).
The existing restore timer runs 600 ms after dispatch, preserving the previous
approximate restore interval without blocking the GUI thread. If clipboard restoration
is disabled, no restore timer is started. The option also applies to SendInput's paste
fallback; successful direct input is unaffected. Logs show `mode=clipboard fast_paste=True`
and the existing phase timings. Disable this experimental option if focus/hotkey
timing makes insertion unreliable in a target application. The alternative clipboard
library checkbox lives in the same group and remains available on every platform;
SendInput and its fallback checkbox are individually hidden on macOS and Linux;
fast paste is hidden only on Linux. Its existing configuration key
`windows_fast_paste` is retained so saved Windows choices remain valid without migration.

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
