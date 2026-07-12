"""SoundPlayer — preload short WAV effects and play them with low latency.

Single responsibility: own the playback PyAudio instance + cached output streams and
play preloaded sounds off the caller's thread. No app state, no Qt.
"""

from __future__ import annotations

import logging
import os
import threading
import wave
from typing import Any, Callable, Dict, List, Optional

import pyaudio


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

    def _pyaudio(self) -> pyaudio.PyAudio:
        if self._pa is None:
            self._pa = pyaudio.PyAudio()
        return self._pa

    def preload(self, names: List[str]) -> None:
        """Load small WAVs fully into memory and pre-open an output stream per format."""
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
        """Low-latency playback of a preloaded short WAV (falls back to on-demand load)."""
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
                stream = self._get_stream(sampwidth, channels, rate)
                if not stream:
                    return
                try:
                    stream.write(frames)
                except Exception as e:
                    logging.error(f"Playback error for '{filename}': {e}")

        threading.Thread(target=_play_cached, daemon=True).start()

    def close(self) -> None:
        """Stop and close all output streams and terminate PyAudio."""
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
