"""Microphone capture: device/stream ownership plus the two PyAudio capture strategies.

* :class:`MicrophoneInput` owns the capture PyAudio instance and the cached input stream,
  opened on the best available device candidate.
* :class:`HotMicCapture` keeps the microphone open (Windows prewarm): a pre-roll buffer avoids
  clipped first words, and stopping retains the block that was being read at that moment.
* :class:`OnDemandCapture` opens the microphone per recording on a reader thread.

No Qt and no app state: callers pass the config accessor and callbacks.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

import pyaudio

from app.audio.device_selector import InputDeviceSelector
from app.core.env import is_MACOS
from app.core.timing import OperationTiming

#: Pause before reopening the stream after a failed read, so a broken driver is not hammered.
_READ_RETRY_S = 0.2
#: Ceiling for waiting on the in-flight read at stop, for a hung audio driver.
_TAIL_WAIT_S = 0.25
#: Pre-roll kept while idle, so the first words before the hotkey are not clipped.
_PREROLL_S = 0.75


class MicrophoneInput:
    """Own the capture PyAudio instance and the cached input stream."""

    def __init__(self, config: Callable[[], Mapping[str, Any]], samplerate: int, chunk_size: int,
                 on_device_fallback: Callable[[str], None]) -> None:
        """Bind the config accessor, the app sample rate and a notifier for device fallbacks."""
        self._config = config
        self.target_samplerate = samplerate
        self.chunk_size = chunk_size
        self._on_device_fallback = on_device_fallback
        self._audio: Optional[pyaudio.PyAudio] = None
        self._stream: Any = None
        self._fallback_notified: Optional[str] = None
        self.samplerate = samplerate
        self.device_index: Optional[int] = None
        self.device_name = ""
        self.devices = InputDeviceSelector(config, samplerate, self.pyaudio)

    def pyaudio(self, refresh: bool = False) -> pyaudio.PyAudio:
        """Return the dedicated capture PyAudio instance, recreating it when ``refresh`` is set."""
        if refresh and self._audio:
            self.terminate()
        if not self._audio:
            self._audio = pyaudio.PyAudio()
        return self._audio

    def terminate(self) -> None:
        """Tear down the capture PyAudio instance."""
        if not self._audio:
            return
        try:
            self._audio.terminate()
        except Exception:
            pass
        self._audio = None

    def open(self) -> Any:
        """Return the cached input stream, opening the first usable device candidate."""
        if self._stream:
            return self._stream
        # macOS: refresh the backend per recording to avoid stale CoreAudio device snapshots
        # after app replacement, permission changes or device switches.
        audio = self.pyaudio(refresh=is_MACOS)
        last_error: Optional[Exception] = None
        for candidate in self.devices.preferred_input_stream_candidates(audio):
            try:
                self._stream = _open_input_stream(audio, candidate, self.chunk_size)
            except Exception as e:
                last_error = e
                logging.warning(
                    "Could not open audio input stream on "
                    f"'{candidate['name']}' (index={candidate['index']}, rate={candidate['samplerate']} Hz): {e}"
                )
                continue
            self.samplerate = candidate["samplerate"]
            self.device_index = candidate["index"]
            self.device_name = candidate["name"]
            logging.info(
                f"Opened audio input stream on '{self.device_name}' "
                f"(index={self.device_index}, rate={self.samplerate} Hz)."
            )
            self._report_fallback(candidate)
            return self._stream

        self.samplerate = self.target_samplerate
        self.device_index = None
        self.device_name = ""
        self.terminate()
        if last_error:
            raise last_error
        raise RuntimeError("No usable audio input device could be opened.")

    def close(self) -> None:
        """Close the cached input stream."""
        if not self._stream:
            return
        for action in (self._stream.stop_stream, self._stream.close):
            try:
                action()
            except Exception:
                pass
        self._stream = None
        if is_MACOS:
            self.terminate()

    def _report_fallback(self, candidate: Mapping[str, Any]) -> None:
        """Notify once per configured device when it could not be opened and the default is used."""
        configured = str(self._config().get("input_device_name", "") or "").strip()
        if not (configured and candidate.get("index") is None):
            # Configured device opened (or none configured): a future fallback may notify again.
            self._fallback_notified = None
            return
        if self._fallback_notified != configured:
            self._fallback_notified = configured
            logging.warning(
                f"Configured input device {configured!r} could not be opened; "
                "recording from the system default instead."
            )
            self._on_device_fallback(configured)


def _open_input_stream(audio: pyaudio.PyAudio, candidate: Mapping[str, Any], chunk_size: int) -> Any:
    """Open a 16-bit input stream for one device candidate (index None = system default)."""
    kwargs: dict[str, Any] = {
        "format": pyaudio.paInt16,
        "channels": candidate["channels"],
        "rate": candidate["samplerate"],
        "input": True,
        "frames_per_buffer": chunk_size,
    }
    if candidate["index"] is not None:
        kwargs["input_device_index"] = candidate["index"]
    return audio.open(**kwargs)


@dataclass
class _PendingStop:
    """A stop request waiting for the read that was in flight when the user stopped."""

    timing: OperationTiming
    frames: list[bytes]
    done: threading.Event = field(default_factory=threading.Event)


class HotMicCapture:
    """Keep the microphone open with a pre-roll buffer; retain the in-flight block at stop."""

    def __init__(self, mic: MicrophoneInput, *, idle_expired: Callable[[], bool],
                 on_block: Callable[[bytes], None]) -> None:
        """Bind the microphone, the idle-timeout check and the level-meter callback."""
        self._mic = mic
        self._idle_expired = idle_expired
        self._on_block = on_block
        self._lock = threading.Lock()
        self._recording = False
        self._frames: list[bytes] = []
        self._preroll: deque[bytes] = deque(maxlen=max(1, int(mic.target_samplerate * _PREROLL_S / mic.chunk_size)))
        self._read_pending = False
        self._pending_stop: Optional[_PendingStop] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    @property
    def running(self) -> bool:
        """Whether the background reader thread is alive."""
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> None:
        """Start the background reader if it is not running yet."""
        if self.running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="HotMicCapture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the reader (joined before the stream closes; closing mid-read segfaults PortAudio)."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None
        self._mic.close()

    def start_recording(self) -> None:
        """Begin a recording that already contains the pre-roll."""
        with self._lock:
            self._frames = list(self._preroll)
            self._recording = True

    def stop_recording(self, timing: OperationTiming) -> Callable[[], list[bytes]]:
        """Stop now and attach the in-flight read; the returned function waits for it and yields the audio."""
        with self._lock:
            self._recording = False
            pending = _PendingStop(timing, self._frames) if self._read_pending else None
            self._pending_stop = pending

        def collect() -> list[bytes]:
            if pending:
                self._wait_for_tail(pending)
            with self._lock:
                return list(self._frames)
        return collect

    def cancel_recording(self) -> None:
        """Stop without attaching the in-flight block."""
        with self._lock:
            self._recording = False

    def _wait_for_tail(self, pending: _PendingStop) -> None:
        """Wait for the in-flight read; on timeout detach it so it can no longer modify the result."""
        timing = pending.timing
        timing.mark("recording_tail_wait_start")
        if not pending.done.wait(_TAIL_WAIT_S):
            with self._lock:
                abandoned = self._pending_stop is pending
                if abandoned:
                    self._pending_stop = None
            if abandoned:
                timing.mark("recording_tail_timeout")
                logging.warning("recording_tail op=%s status=timeout; pending audio could not be retained",
                                timing.operation_id)
        timing.mark("recording_tail_wait_end")

    def _loop(self) -> None:
        """Read continuously, feeding the pre-roll and, while recording, the recording."""
        logging.info("Background audio capture thread started.")
        while self._running:
            if not self._recording and self._idle_expired():
                logging.info("Stopping background audio capture due to Windows prewarm idle timeout.")
                self._running = False
                break
            try:
                stream = self._mic.open()
                with self._lock:
                    self._read_pending = True
                data = stream.read(self._mic.chunk_size, exception_on_overflow=False)
                self._on_block(data)
                with self._lock:
                    self._preroll.append(data)
                    pending = self._pending_stop
                    if pending is None and self._recording:
                        self._frames.append(data)
                if pending is not None:
                    self._retain_tail(stream, data, pending)
            except Exception as e:
                logging.warning(f"Background audio capture error: {e}")
                with self._lock:
                    pending, self._pending_stop = self._pending_stop, None
                    self._read_pending = False
                if pending:
                    pending.timing.mark("recording_tail_read_failed")
                    pending.done.set()
                self._mic.close()
                time.sleep(_READ_RETRY_S)
            finally:
                with self._lock:
                    self._read_pending = False
                    pending, self._pending_stop = self._pending_stop, None
                if pending:
                    pending.done.set()
        self._mic.close()
        logging.info("Background audio capture thread finished.")

    def _retain_tail(self, stream: Any, data: bytes, pending: _PendingStop) -> None:
        """Append the stopped block plus already buffered samples, unless the stop timed out."""
        timing = pending.timing
        timing.mark("recording_tail_read_end")
        # Read only samples already buffered; never add a fixed post-recording sleep.
        buffered = b""
        try:
            available = stream.get_read_available()
            if available > 0:
                buffered = stream.read(available, exception_on_overflow=False)
        except Exception:
            timing.mark("recording_tail_drain_failed")
        with self._lock:
            saved = self._pending_stop is pending
            if saved:
                pending.frames.append(data)
                if buffered:
                    pending.frames.append(buffered)
                    self._preroll.append(buffered)
                timing.mark("recording_tail_saved")
        logging.info("recording_tail op=%s retained=%s pending_frames=%s buffered_frames=%s samplerate=%s",
                     timing.operation_id, saved, len(data) // 2, len(buffered) // 2, self._mic.samplerate)


class OnDemandCapture:
    """Open the microphone per recording and read it on a dedicated thread."""

    def __init__(self, mic: MicrophoneInput, *, on_block: Callable[[bytes], None],
                 on_failure: Callable[[], None]) -> None:
        """Bind the microphone, the level-meter callback and the open-failure handler."""
        self._mic = mic
        self._on_block = on_block
        self._on_failure = on_failure
        self._recording = False
        self._frames: list[bytes] = []
        self._thread: Optional[threading.Thread] = None

    def start_recording(self) -> None:
        """Start reading on a new thread."""
        self._frames = []
        self._recording = True
        self._thread = threading.Thread(target=self._read, name="OnDemandCapture", daemon=True)
        self._thread.start()

    def stop_recording(self, _timing: OperationTiming) -> Callable[[], list[bytes]]:
        """Stop now; the returned function waits for the reader and yields the audio."""
        self._recording = False

        def collect() -> list[bytes]:
            self._join()
            return list(self._frames)
        return collect

    def cancel_recording(self) -> None:
        """Stop and wait for the reader."""
        self._recording = False
        self._join()

    def _join(self) -> None:
        """Wait for the reader thread to finish its last read."""
        if self._thread and self._thread.is_alive():
            self._thread.join()

    def _read(self) -> None:
        """Read blocks until the recording stops."""
        logging.info("Audio recording thread started (PyAudio).")
        try:
            stream = self._mic.open()
        except Exception as e:
            logging.error(f"Could not open audio input stream: {e}")
            self._recording = False
            self._on_failure()
            return
        if self._mic.device_name:
            logging.info(
                f"Recording from input device '{self._mic.device_name}' "
                f"(index={self._mic.device_index}, rate={self._mic.samplerate} Hz)."
            )
        while self._recording:
            try:
                data = stream.read(self._mic.chunk_size, exception_on_overflow=False)
            except Exception as e:
                logging.error(f"Error while reading audio stream: {e}")
                break
            self._on_block(data)
            self._frames.append(data)
        self._mic.close()
        logging.info("Audio recording thread finished.")
