"""SoundPlayer — preload short WAV effects and play them with low latency.

Single responsibility: own the playback PyAudio instance + cached output streams and
play preloaded sounds off the caller's thread. No app state, no Qt.

Windows: once the audio output has been idle for about 20 s, the next sound blocks the
microphone's MME read in the same (windowless) process for ~0.7 s, which delays the
recording feedback and the stop sound. A keepalive writes a little silence to the open
output streams every few seconds, so the output never falls idle.
"""

from __future__ import annotations

import logging
import os
import threading
import wave
from typing import Any, Callable, Dict, List, Optional

import pyaudio

from app.core.env import is_WINDOWS

#: Pause between two silence writes of the Windows keepalive, and the length of each.
_KEEPALIVE_INTERVAL_S = 5.0
_KEEPALIVE_SILENCE_S = 0.02
#: Longest wait of close() for a running write (sounds are a few hundred milliseconds).
_CLOSE_WAIT_S = 2.0


class SoundPlayer:
    """Preload short WAV effects and play them with low latency."""

    def __init__(self, resource_resolver: Callable[..., str]) -> None:
        """Store the resource resolver used to locate sound files."""
        # resource_resolver(*segments) -> absolute path (e.g. app.core.paths.resource_path)
        self._resolve = resource_resolver
        self._pa: Optional[pyaudio.PyAudio] = None
        self._cache: Dict[str, tuple] = {}
        self._streams: Dict[tuple, Any] = {}
        # Serializes stream writes: each play() runs on its own daemon thread, and two
        # overlapping sounds with the same format would otherwise write to one shared
        # stream concurrently (PyAudio streams are not documented as thread-safe).
        self._play_lock = threading.Lock()
        self._names: List[str] = []
        self._enabled = True
        self._keepalive: Optional[threading.Thread] = None
        self._keepalive_stop = threading.Event()

    def _pyaudio(self) -> pyaudio.PyAudio:
        if self._pa is None:
            self._pa = pyaudio.PyAudio()
        return self._pa

    def set_enabled(self, enabled: bool) -> None:
        """Turn the sounds on or off; off plays nothing and keeps no output stream open."""
        if enabled == self._enabled:
            return
        self._enabled = enabled
        if enabled:
            self.preload(self._names)
        else:
            self.close()
        logging.info("Start/stop sounds %s.", "enabled" if enabled else "disabled")

    def preload(self, names: List[str]) -> None:
        """Load small WAVs fully into memory and pre-open an output stream per format."""
        self._names = list(names)
        if not self._enabled:
            return
        try:
            self._pyaudio()
        except Exception as e:
            logging.warning(f"PyAudio init failed (sounds disabled): {e}")
            return
        for name in names:
            path = self._resolve("resources", name)
            if not os.path.isfile(path):
                logging.debug(f"Sound file missing (skip preload): {path}")
                continue
            try:
                with wave.open(path, "rb") as wf:
                    self._cache[name] = (
                        wf.getsampwidth(), wf.getnchannels(),
                        wf.getframerate(), wf.readframes(wf.getnframes()),
                    )
            except Exception as e:
                logging.warning(f"Failed to preload {name}: {e}")
        for sampwidth, channels, rate, _frames in self._cache.values():
            try:
                self._get_stream(sampwidth, channels, rate)
            except Exception as e:
                logging.error(f"Failed to pre-open stream for ({sampwidth}, {channels}, {rate}): {e}")
        if is_WINDOWS and self._streams:
            self._start_keepalive()

    def _start_keepalive(self) -> None:
        # A keepalive that close() already stopped may still be winding down; it never writes again.
        if self._keepalive is not None and self._keepalive.is_alive() and not self._keepalive_stop.is_set():
            return
        self._keepalive_stop = threading.Event()
        self._keepalive = threading.Thread(target=self._keep_output_awake, args=(self._keepalive_stop,),
                                           name="SoundKeepalive", daemon=True)
        self._keepalive.start()

    def _keep_output_awake(self, stop: threading.Event) -> None:
        """Write a short silence to every open output stream until ``stop`` is set.

        After ~20 s without output, the first sound of a windowless process stalls the warm
        microphone's MME read for ~0.7 s; output that never idles avoids the stall.
        """
        while not stop.wait(_KEEPALIVE_INTERVAL_S):
            with self._play_lock:
                if stop.is_set():
                    return
                for (sampwidth, channels, rate), stream in list(self._streams.items()):
                    try:
                        stream.write(b"\x00" * (int(rate * _KEEPALIVE_SILENCE_S) * sampwidth * channels))
                    except Exception as e:
                        logging.debug(f"Sound keepalive write failed: {e}")

    def _get_stream(self, sampwidth: int, channels: int, rate: int) -> Optional[Any]:
        key = (sampwidth, channels, rate)
        if key in self._streams:
            return self._streams[key]
        try:
            pa = self._pyaudio()
        except Exception as e:
            logging.error(f"PyAudio init failed for playback: {e}")
            return None
        try:
            stream = pa.open(
                format=pa.get_format_from_width(sampwidth),
                channels=channels, rate=rate, output=True,
            )
            self._streams[key] = stream
            return stream
        except Exception as e:
            logging.error(f"Could not open output stream: {e}")
            return None

    def play(self, filename: str) -> None:
        """Low-latency playback of a preloaded short WAV (falls back to on-demand load); nothing when off."""
        if not self._enabled:
            return

        def _play_cached() -> None:
            data_tuple = self._cache.get(filename)
            if not data_tuple:
                path = self._resolve("resources", filename)
                if not os.path.isfile(path):
                    logging.debug(f"Sound file not found: {path}")
                    return
                try:
                    with wave.open(path, "rb") as wf:
                        data_tuple = (
                            wf.getsampwidth(), wf.getnchannels(),
                            wf.getframerate(), wf.readframes(wf.getnframes()),
                        )
                        self._cache[filename] = data_tuple
                except Exception as e:
                    logging.error(f"Failed to load sound '{filename}': {e}")
                    return
            sampwidth, channels, rate, frames = data_tuple
            with self._play_lock:
                if not self._enabled:
                    return
                stream = self._get_stream(sampwidth, channels, rate)
                if not stream:
                    return
                try:
                    stream.write(frames)
                except Exception as e:
                    logging.error(f"Playback error for '{filename}': {e}")

        threading.Thread(target=_play_cached, daemon=True).start()

    def close(self) -> None:
        """Stop the keepalive, close all output streams and terminate PyAudio.

        Runs on the GUI thread. A blocking MME write never gives up on an output device that
        vanished, so a write that still holds the lock is not waited for: its streams are left
        alone (closing them under the write would crash PortAudio) and no sound plays any more.
        """
        self._keepalive_stop.set()
        if not self._play_lock.acquire(timeout=_CLOSE_WAIT_S):
            logging.warning("Sound output did not finish a write within %.0f s; leaving its streams open.",
                            _CLOSE_WAIT_S)
            self._enabled = False
            return
        try:
            for s in self._streams.values():
                try:
                    s.stop_stream()
                    s.close()
                except Exception:
                    pass
            self._streams.clear()
            if self._pa is not None:
                try:
                    self._pa.terminate()
                except Exception:
                    pass
                self._pa = None
        finally:
            self._play_lock.release()
