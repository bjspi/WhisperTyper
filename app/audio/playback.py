"""WavPlayer — play one WAV file in the app, stoppable at any time.

Single responsibility: stream a WAV to the default output device on a background thread.
No Qt; the caller turns the completion callback into a GUI-thread update.
"""
from __future__ import annotations

import logging
import threading
import wave
from typing import Any, Callable, Optional

import pyaudio

#: Frames written per chunk; small enough that a stop takes effect within ~0.1 s.
_CHUNK_FRAMES = 1024


class WavPlayer:
    """Play a WAV file on a background thread; a new ``play`` or ``stop`` ends the previous one."""

    def __init__(self, audio_factory: Callable[[], Any] = pyaudio.PyAudio) -> None:
        """``audio_factory`` creates the PyAudio instance on first use."""
        self._audio_factory = audio_factory
        self._audio: Optional[Any] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def play(self, path: str, on_finished: Callable[[], None]) -> None:
        """Start playing ``path``; ``on_finished`` runs on the playback thread when it ends or stops."""
        self.stop()
        stop = self._stop = threading.Event()

        def run() -> None:
            try:
                self._stream(path, stop)
            except Exception as error:
                logging.warning("recording_playback path=%s error=%s", path, error)
            finally:
                on_finished()

        self._thread = threading.Thread(target=run, name="RecordingPlayback", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the current playback (no-op when nothing plays)."""
        self._stop.set()

    def is_playing(self) -> bool:
        """True while a playback runs and has not been asked to stop."""
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def _stream(self, path: str, stop: threading.Event) -> None:
        """Write the file to an output stream chunk by chunk until it ends or ``stop`` is set."""
        if self._audio is None:
            self._audio = self._audio_factory()
        audio = self._audio
        with wave.open(path, "rb") as wav:
            stream = audio.open(format=audio.get_format_from_width(wav.getsampwidth()),
                                channels=wav.getnchannels(), rate=wav.getframerate(), output=True)
            try:
                while not stop.is_set():
                    frames = wav.readframes(_CHUNK_FRAMES)
                    if not frames:
                        break
                    stream.write(frames)
            finally:
                stream.stop_stream()
                stream.close()
