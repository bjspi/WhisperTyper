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
| `stop_to_first_byte_ms` | Stop detection → HTTP response status/headers received |
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
| `audio_encode_ms` | Read the prepared WAV and encode AAC/M4A with PyAV in the upload worker (AAC only) |
| `audio_compress_ms` | FFmpeg compression of a recording that exceeds the upload limit (long recordings only) |
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

## Recording start: hotkey → microphone delivers audio

Each start gets its own operation with `source=recording_start`, from the hotkey listener's
detection to the moment the recording feedback (balloon or prompt palette) is shown. Its
`latency_summary` has `outcome=ok`, or `stopped_before_audio`, `cancelled_before_audio`,
`capture_failed`, `not_configured`; the following `Recording started (op=…, capture=hot_mic|on_demand|macos,
stream_was_open=…)` line tells whether a warm microphone was already reading.

| Field | Interval |
| --- | --- |
| `start_event_queue_ms` | Hotkey listener detection → start handler on the GUI thread |
| `start_checks_ms` | Configuration and permission checks, connection warm-up scheduling |
| `start_state_ui_ms` | Recording-state signal: tray icon and everything else that reacts to it |
| `capture_request_ms` | Ask the capture to record (start the reader thread / flag the warm microphone) |
| `selection_context_ms` | Copy the selected text as rephrasing context (simulated Ctrl+C and clipboard wait) |
| `start_sound_ms` | Start sound dispatch |
| `start_handler_ms` | Whole start handler on the GUI thread |
| `mic_thread_wait_ms` | Capture request → the reader thread runs (only when a thread had to start) |
| `pyaudio_init_ms` | PortAudio start-up (first recording, or after the warm microphone was released) |
| `device_candidates_ms` | Choose the input device candidates |
| `stream_open_ms` | Open the input stream, including failed candidates |
| `device_name_ms` | Resolve the device's full name (may enumerate all devices once) |
| `first_block_read_ms` | First read until the first block arrives (on demand; 64 ms per 1,024 samples at 16 kHz) |
| `hotkey_to_first_audio_ms` | Hotkey → the first recorded block |
| `audio_to_feedback_queue_ms` | First block (capture thread) → GUI thread handles it |
| `feedback_ms` | Show the recording balloon or the prompt palette |
| `hotkey_to_feedback_ms` | Hotkey → the recording feedback is shown |

With the warm microphone and an open stream there are no stream phases, and the take
already holds the pre-roll from before the hotkey. Without audio after 1.5 s the
`no_audio_grace_elapsed` milestone is logged.

For the warm microphone, `hotmic_start op=… last_block_age_ms=… read_in_flight_ms=…
reader_alive=…` shows at each start whether the reader is alive, when its last block arrived
and how long the current driver read has been running. Every pause of more than 150 ms between
two blocks logs `hotmic_gap gap_ms=… read_ms=… outside_read_ms=… available_before=…
recording=…`: `read_ms` is time inside the driver read, `outside_read_ms` time in Python between
reads, `available_before` the frames the driver held before the read.

On Windows, a sound played after about 20 s of silent audio output used to block the
microphone's MME read in the windowless app for ~0.7 s (`hotmic_gap read_ms≈700
outside_read_ms=0`), which delayed the recording feedback and the stop sound. While the
start/stop sounds are on, a keepalive therefore writes 20 ms of silence to the open output
streams every 5 s. With **Play sounds at start and stop** off, no output stream is opened and
no keepalive runs.

## Recording boundary on Windows

With **Keep mic hot**, stopping a recording attaches the active input read to
that recording before disabling capture. The capture thread retains the in-flight
block and reads a snapshot of already available driver-buffered samples before the
WAV is written. There is no fixed post-recording sleep and no wait if a read is not
active. At 16 kHz, the normal 1,024-sample block spans 64 ms, so dropping the in-flight
read would lose up to that much speech at the end. Finishing that read can naturally
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
| `prepare_ms` | API call start → request hook, including multipart/JSON assembly and proxy routing |
| `tcp_ms` | DNS resolution + TCP setup (httpcore resolves inside its connect step) |
| `proxy_tls_ms` / `proxy_tunnel_ms` | TLS to an HTTPS proxy / CONNECT tunnel setup |
| `tls_setup_ms` | Origin TLS setup: context/CA preparation, handshake and certificate verification |
| `connect_ms` | Entire connection setup including DNS/TCP/TLS and any proxy tunnel |
| `ttfb_ms` / `after_upload_wait_ms` | Request headers sent / complete request sent → response status and headers received |
| `upload_ms` | First body send → complete request sent; multipart framing is included |
| `request_to_first_byte_ms` | API call start → response headers, including preparation and connection setup |
| `response_body_ms` | Complete headers → complete consumed/decompressed response body |
| `total_ms` | API call start → complete response or failure |
| `connection` / `reused` | Socket identity / whether this exchange reused an existing socket |
| `error` / `last_phase` | Exception class (without private exception text) / last milestone reached before completion or failure |
| `upload_bytes` / `response_body_bytes` | Request `Content-Length` / raw response body bytes received |

Transport milestones also appear as `latency_event` phases such as
`transcription_http_request_sent` and `transcription_http_first_byte` for operations
with timing state. Use their captured `at_ms` to determine when the POST finished
sending, rather than the log writer's line date. New connections have TCP/TLS
fields; reused ones omit them (`reused=True` means no connect step in this exchange,
also for HTTPS). `tls_setup_ms` intentionally includes certificate verification,
rather than pretending to isolate only the cryptographic handshake.
All milestones come from httpcore's documented `trace` extension and httpx event
hooks; no transport internals are subclassed. That API reports the parsed status
line and headers, not the first raw socket byte, so `first_byte` marks "response
headers received" (providers send status and headers together in practice), and
DNS is not measured separately from TCP. Sending completion means the OS accepted
the bytes, not that the server acknowledged reading all of them.
`after_upload_wait_ms` combines transit and provider processing; the client cannot
split the provider's queue from inference without server-side timing data.

## Persistent connections and background warming

All transcription, rephrasing and connection-test calls share pooled httpx clients
(one thread-safe client per proxy route, HTTP/1.1), even when each call uses a new
Qt worker thread or a different Groq key. Headers are per request and no cookies are
stored. TLS verification (certifi, `SSL_CERT_FILE`/`SSL_CERT_DIR`, and a legacy
`REQUESTS_CA_BUNDLE`), explicit/px proxies and system/environment proxies including
`NO_PROXY` remain active. TCP_NODELAY and OS TCP keepalive are enabled; idle pooled
connections are discarded after 60 s, because routers drop idle connections silently and
reusing a dead one would hang until the read timeout; a warm-up that still hits a dead
connection is repeated once on a fresh one. Failed paid POSTs are not retried. Busy pools open
another connection instead of waiting for a warm-up request.

A daemon performs auth-free HEAD requests at startup, after saving settings and
when recording begins. Selecting a rephrasing prompt in the recording palette
also triggers warming immediately and puts the configured rephrasing endpoint
(including OpenAI) first, before the transcription endpoint. The selection
reactivates the five-minute activity window; selecting Standard adds no request.
While recording or for five minutes after recent activity,
a timer refreshes each configured origin at most every 20 seconds. Groq/OpenAI
use their `/models` path; custom endpoints use the origin root. URL credentials,
queries, API keys and audio are excluded; httpx never adds netrc credentials. Redirects are
disabled. A 401/404/405 still proves that the connection reached the server.
Warm-up has short timeouts and never delays capture, upload or shutdown waiting
for completion. The server can still close idle connections; httpcore detects that
and reconnects. There is no guarantee of reuse during overlapping requests.

## Parallel second attempt on a stalled connection

With **Start a parallel second attempt when the connection stalls** (API Providers tab), a
request that stalls is raced by a second one. A pooled connection that lost a packet waits in
TCP retransmission backoff (about 1 s, 2 s, 4 s, …), which can hold an upload for several
seconds even though the network is fine again; a fresh connection has no backoff.

The second attempt starts when the upload is not finished after
`max(0.5 s, upload size at 1.5 MB/s)`, or when no response byte has arrived
`wait + 0.5 s per uploaded MB` after the upload finished; the wait is the setting
**Wait for the response before the 2nd attempt** below the option (default 1.0 s). "Upload
finished" means the body was handed to the operating system's send buffer, so a small
recording counts as uploaded at once and a stall then shows up as a missing response. The
other thresholds are named constants in `app/services/transcription.py`.

An endpoint on this computer (`localhost`, `*.localhost`, `127.0.0.0/8`, `::1`) is never
hedged: it has no network path to stall on, and a second request would only double the
local server's load. Such endpoints are also always reached without a proxy, whatever the
proxy settings say (a proxy would resolve `localhost` as itself).

The second attempt always opens a new connection (DNS, TCP, TLS) and uses the next Groq key
when rotation is on, otherwise the same key. The first successful response wins and is
processed exactly once — replacements, rephrasing or the instruction, insertion; the other
attempt finishes in the background and is discarded. A failure that arrives before any stall
is reported exactly as without the option; when one attempt fails after the hedge started,
the other is awaited, and if both fail the first error is shown. Without the option a paid
request is never sent twice.

Logs of a hedged request, in order:
- `transcription_hedge op=… started trigger=upload|response after_ms=… upload_bytes=…
  fresh_connection=true` when the second attempt starts;
- `transcription_hedge op=… attempt=1|2 finished_ms=… outcome=ok|<error class>` for each
  attempt when it ends (measured from the first attempt's start), including the discarded one;
- `transcription_hedge op=… trigger=… winner=1|2|none winner_ms=…` for the result used;
- the `transcription_hedge_started` and `transcription_hedge_won_<n>` milestones,
  `hedge_trigger_ms` in the summary, and an `http_transport` line with
  `stage=transcription_hedge` (connection setup, upload, response) for the second attempt.

## Transcription confidence and the quality fallback

**Log transcription confidence** (General tab, Logging; on by default) asks the API for its
confidence details where the model offers them: Whisper models (Groq, `whisper-1`) answer
with `verbose_json` (average log-probability, no-speech probability and compression ratio
per segment, timestamps per word), OpenAI's GPT transcription models with a log-probability
per token, grouped into words. Log-probabilities are natural logs: 0 is certain, more
negative is less sure. The transcribed text and the latency stay the same.

- `transcription_diagnostics op=… source=verbose_json|logprobs … avg_logprob=…` sums up
  every transcription (no transcript text);
- with debug logging, one line per segment follows, and the word list when log redaction
  is off;
- "Mark last recording as bad" stores the full details in the JSON sidecar.

**Quality fallback** (Transcription tab, below the prompt) adds a second transcription with
its own prompt (empty = no prompt), for the rare results a long prompt garbles (dropped or
rewritten text, typically an average log-probability far below -0.5 while normal dictations
sit around -0.1 to -0.3). *Only for doubtful results* (default) sends it after an empty
result or one below the threshold; *Always, in parallel* sends both at once and waits for
the second only when the first is doubtful or fails. In both modes the first result is kept
unless it is doubtful and the second one is more confident. The second request has its own
operation id and no hedge, so the first request's timings stay comparable.

The fallback needs confidence data and is skipped without it:
- models that offer none (any model name not listed for Whisper/GPT transcription, e.g. on a
  custom or local server) do not even get the second request; the log says
  `transcription_second_pass op=… skipped reason=no_confidence_data`;
- a response without a `segments` list and without a `logprobs` list, or with an
  `avg_logprob` of exactly 0, counts as "no data" and is used as it is, even when empty.
A local server that cannot compute log-probabilities should simply leave these fields out.

- `transcription_second_pass op=… started mode=… second_op=… prompt=…`;
- `transcription_second_pass op=… finished avg_logprob=… words=…` or `… failed error=…`;
- `transcription_second_pass op=… chosen=first|second reason=confident|first_doubtful|first_failed|second_failed …`.

The checkmark balloon shows the average log-probability and whether the second
transcription or a second connection was used, in brackets: green, yellow when the second
transcription was used or the value is below 70% of the threshold, red below the threshold.

## Connection: IPv4 or IPv6

**Connection** (General tab) chooses the IP version of every API connection, including
warm-up, connection tests and second attempts: *Automatic* leaves it to the operating
system, *IPv4 only* and *IPv6 only* bind new connections to that address family. Changing it
closes the pooled connections. *IPv4 only* helps when the IPv6 route loses packets, which
shows up as recurring hedges, long `connect_ms` or `upload_ms` values and losses in a ping
over IPv6 (`ping -6 api.groq.com`) but not over IPv4 (`ping -4 api.groq.com`). A proxy
must be reachable over the chosen IP version.

HTTP and HTTPS proxies (including CONNECT tunnels) receive full measurements. SOCKS
proxies are not supported by the default install (httpx needs the optional
`socksio` package).

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
fast paste is hidden only on Linux. Its configuration key is `fast_paste`; a choice saved
under the former `windows_fast_paste` key is migrated on load.

## Recording formats

Transcription → Recording offers **WAV (PCM)** and **AAC (M4A)**. WAV is the
default: mono 16-bit PCM at 16 kHz (256 kbit/s). AAC is encoded with
[PyAV](https://pyav.basswood-io.com/), whose wheels bundle the FFmpeg libraries, so
Windows, macOS and Linux share one code path and no external FFmpeg executable is
needed. The prepared WAV is streamed in 100 ms blocks, resampled to 48 kHz mono and
written as AAC-LC in an MP4/M4A container at 48, 64, 96, 128, 160 or 192 kbit/s
(default 64 kbit/s). A background probe after startup checks once that PyAV and its
AAC encoder load; if not, only WAV is offered. OPUS is excluded because
transcription providers do not accept it consistently.

Capture, final-block retention, minimum-duration checks, gain and resampling
remain the same for both formats. The prepared WAV stays available for playback
and retries. AAC is generated after stop in the existing transcription worker,
so encoding does not block the GUI. The temporary M4A is uploaded with
`audio/mp4` and deleted on success or failure. `audio_encode_ms` measures
encoding, including reading the prepared WAV. It is part of `upload_prepare_ms`
and the existing stop-to-response/output totals; do not add overlapping fields.
`recording_upload` logs the operation ID, selected format/bitrate, WAV source size
and upload size at INFO, without audio or credentials.

Retained recordings, including retry/retranscription, use the current saved
recording format. An encoder error keeps the WAV and reports a normal retryable
error. Only a recording still over the configured upload limit (with 16 kHz WAV
roughly 13 minutes at the default 24 MB) resolves FFmpeg and compresses the
retained WAV exactly like a picked file; `audio_compress_ms` measures that step.
Without FFmpeg such a recording is rejected with a suggestion to use AAC/a lower
bitrate, install FFmpeg or record a shorter clip.
Picked files/videos keep their existing extraction/compression behavior.

## Logging does not wait for the disk

All application logging, including the timing records, runs through the standard
library's `QueueHandler`/`QueueListener` (`app/core/log_queue.py`), installed during
bootstrap before recording/hotkeys. Producers only enqueue the unformatted record;
formatting (including the lazily computed `latency_summary`/`http_transport`
durations), handler locks, rotation and file writes run on the listener thread.
Output handlers are swapped copy-on-write, so adding or removing the file handler
never waits for the listener; closing a removed handler is queued behind its records.
No synchronous flush is performed during recording, upload or result delivery.
Timing milestones are ordinary records on the `whispertyper.timing` logger
(`app/core/timing.py`), so they follow the configured log level and sinks.

Creating a log record still costs a few microseconds; this does not promise zero
overhead. The queue is unbounded so producers do not wait: a persistently blocked
sink can grow memory usage. At interpreter exit the listener drains the queue.
Because Qt aborts the process after an uncaught exception in a slot (skipping
`atexit`), `sys.excepthook` and `threading.excepthook` flush the queue first, so the
records leading up to a crash reach the log file. Only a hard native crash
(e.g. a segfault in a driver) can still lose the last queued records. A failing
sink never stops the other sinks or the transcription pipeline.

## Other diagnostics in the log

- **Native crashes:** a crash inside PortAudio or Qt ends the process before Python can log
  it. The Python stacks of all threads are then written to `WhisperTyper.crash.log` in the
  app data folder (`~/.WhisperTyper`), and the next start logs `The previous session crashed
  natively` with the last lines of that trace.
- **Qt's own messages** (stylesheet errors, window geometry, plugins) are logged as
  `qt category=… …` instead of going to the invisible stderr of a windowless launch.
- **Saved settings:** `settings_saved changes: key=old->new …` lists every setting a save
  changed. Credentials, proxy URLs, prompts, prompt entries and replacements appear only as
  `key=changed`; endpoint URLs by host only.
- **Prompt palette:** `prompt_palette shown prompts=… preselected=… position=… opacity=…`,
  `prompt_palette selected=<caption> kind=…` (the caption is redacted like other user text)
  and `prompt_palette closed prompt=chosen|automatic|none`.
