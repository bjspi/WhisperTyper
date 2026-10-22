# Local transcription

WhisperTyper can transcribe on your own computer instead of a cloud API. It is optional, off by
default, and available on Windows and Linux (switched off on macOS).

## Turning it on

1. **Settings → Local transcription → Enable local transcription**, or pick **Local (this PC)**
   as the provider on the Transcription page.
2. The first time, WhisperTyper lists the Python packages it needs and asks before installing
   them with pip into the Python environment it runs in (the project venv when there is one).
   It installs about 300 MB, or about 2 GB when an NVIDIA GPU is present (the CUDA libraries).
   pip's output shows live. If the install fails, the switch stays off and the output is
   also written to `WhisperTyper.log`.
3. The model is downloaded from Hugging Face the first time the engine loads (0.5–3 GB). The
   status line shows the progress.

By hand: `pip install -e ".[local]"`, plus `.[local-nvidia]` for NVIDIA or `.[local-intel]` for Intel GPUs/NPUs.

## Devices and backends

| Device | Backend | Default model |
|---|---|---|
| NVIDIA GPU | faster-whisper / CTranslate2, CUDA float16 | `mobiuslabsgmbh/faster-whisper-large-v3-turbo` |
| Intel GPU (Arc, Iris Xe, Core Ultra), Intel NPU | OpenVINO GenAI | `OpenVINO/whisper-large-v3-turbo-int4-ov` |
| CPU | CTranslate2 int8 (or OpenVINO CPU) | as above |

**Automatic** picks the fastest device that is present: CUDA, then Intel GPU, then CPU. OpenVINO's
GPU plugin only drives Intel GPUs, so NVIDIA cards use the CTranslate2 backend.

## When it runs

The engine runs only while it is enabled **and** needed: the provider is **Local**, or the
server is on.

- **Preload at startup** (default): loads when WhisperTyper starts.
- **On the first recording**: loads when a recording starts, while you are still speaking.
- **Unload when idle**: frees the GPU memory after N minutes without use; the next recording
  loads it again. This does not apply while the server is on.
- With the provider **Local** there is no model, temperature, key or quality fallback to
  configure; the engine reports no confidence data, so the quality fallback never runs.

## Server for other programs

**Provide an OpenAI-compatible endpoint** keeps the engine loaded on a fixed address and port
(default `127.0.0.1:9017`):

```text
POST http://127.0.0.1:9017/v1/audio/transcriptions   multipart: file, [language], [prompt], [response_format]
GET  http://127.0.0.1:9017/v1/models
GET  http://127.0.0.1:9017/health                     ?deep=1 runs a tiny transcription
```

Any address other than `127.0.0.1` makes the server reachable from the network **without
authentication**.

## Self-healing

A watchdog thread checks `/health` every 10 seconds. Every 15 minutes, and right after standby,
it also runs a tiny inference, which catches a GPU that hangs after resume. It restarts the
engine after a crash, a hang or a model error (backoff 0 → 300 s) and stops engines left over by
a crashed app.

## Logs for support

- `~/.WhisperTyper/local_engine.log` is the engine process log. It contains the Python and
  package versions, the detected devices, the settings, load times, every request (language,
  audio and speech length, decode/VAD/inference time) and full tracebacks. With **debug
  logging** on, it logs at DEBUG level. With redaction off, it also contains the transcribed text.
- `~/.WhisperTyper/WhisperTyper.log` has the supervisor's states, the start/stop decisions
  (`local_engine_*`), the device detection, the pip output and the requests the app sent.

Open them under **Settings → Local transcription → Open engine log** or from the tray menu.

## Models

Models are stored in `~/.WhisperTyper/local_models` unless you choose another folder. The folder
gets a `.nobackup` marker. **Delete…** next to the model removes a downloaded model, unloading
the engine first if it is in use.
