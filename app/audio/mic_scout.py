"""First-run help: find a microphone that hears the user while the system default stays silent.

A laptop's built-in microphone can be the system default and deliver nothing while the headset
is the one that works. Until the default microphone was heard once, a take is watched: when it
stays silent for a moment, the other input devices are opened alongside it, and the first one
that clearly hears the user is reported. The scout only measures levels; no audio is kept.

Opening a Bluetooth headset's microphone can switch its playback to the hands-free profile for
a moment, which is why the other devices stay closed until the default has proven silent.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from app.audio.capture import MicrophoneInput, open_input_stream
from app.core import dsp

#: A block peak at or above this level counts as a voice rather than background noise.
SIGNAL_DBFS = -45.0
#: So many loud blocks make a signal; a single click or bump does not.
MIN_SIGNAL_BLOCKS = 3
#: Another device must be this much louder than anything the default delivered so far.
SWITCH_MARGIN_DB = 20.0
#: The other devices stay closed for this long, so a short pause never opens them.
SILENCE_GRACE_S = 2.5
#: The other devices are listened to for at most this long per take.
LISTEN_WINDOW_S = 30.0
_POLL_S = 0.05
_SILENCE_DBFS = -96.0
_FULL_SCALE = 32768.0
#: Name parts of inputs that capture playback rather than a voice (they "hear" any music).
_LOOPBACK_NAME_PARTS = ("stereo mix", "stereomix", "mezcla est", "mixage st", "what u hear", "wave out",
                        "loopback", "cable output", "voicemeeter", "vb-audio")


@dataclass(frozen=True)
class ScoutDevice:
    """An input device worth listening to."""

    index: int
    name: str  # the value stored as ``input_device_name``
    label: str  # the full name for showing


def scout_devices(devices: List[Dict[str, Any]], current_name: str) -> List[ScoutDevice]:
    """The real microphones among ``devices``, without the one already being recorded from."""
    candidates = []
    for device in devices:
        label = str(device.get("label", device["name"]))
        if current_name in (device["name"], label):
            continue
        if any(part in label.lower() for part in _LOOPBACK_NAME_PARTS):
            continue
        candidates.append(ScoutDevice(int(device["index"]), str(device["name"]), label))
    return candidates


def _peak_dbfs(pcm: bytes) -> float:
    """Peak level of a 16-bit PCM block in dBFS."""
    peak = dsp.peak(pcm)
    return 20.0 * math.log10(peak / _FULL_SCALE) if peak > 0 else _SILENCE_DBFS


@dataclass
class _Listener:
    """One opened device and the loud blocks heard on it so far."""

    device: ScoutDevice
    stream: Any
    loud_blocks: int = 0


class MicrophoneScout:
    """Watches one take on its own daemon thread; the callbacks arrive on that thread.

    ``on_heard`` reports that the microphone in use delivered a signal, ``on_found`` that another
    device hears the user while it does not. At most one of them is called, once.
    """

    def __init__(self, mic: MicrophoneInput, *, on_heard: Callable[[], None],
                 on_found: Callable[[ScoutDevice], None]) -> None:
        """``mic`` is the input being recorded from; the scout reads its device list and settings."""
        self._mic = mic
        self._on_heard = on_heard
        self._on_found = on_found
        self._stop = threading.Event()
        self._current_peak_db = _SILENCE_DBFS
        self._current_loud_blocks = 0
        self._thread = threading.Thread(target=self._run, name="MicrophoneScout", daemon=True)

    def start(self) -> None:
        """Begin watching."""
        self._thread.start()

    def stop(self) -> None:
        """End the search without waiting: the thread may sit in a slow device call."""
        self._stop.set()

    def observe(self, pcm: bytes) -> None:
        """Take in a block from the microphone being recorded from (capture thread)."""
        level = _peak_dbfs(pcm)
        self._current_peak_db = max(self._current_peak_db, level)
        if level >= SIGNAL_DBFS:
            self._current_loud_blocks += 1

    def _current_heard(self) -> bool:
        """Whether the microphone in use has delivered a signal."""
        return self._current_loud_blocks >= MIN_SIGNAL_BLOCKS

    def _run(self) -> None:
        """Wait out the grace period, then listen on the other devices if nothing was heard yet."""
        deadline = time.monotonic() + SILENCE_GRACE_S
        while time.monotonic() < deadline and not self._current_heard() and not self._stop.wait(_POLL_S):
            pass
        found = None
        if not self._current_heard() and not self._stop.is_set():
            found = self._search()
        if found is not None:
            self._on_found(found)
        elif self._current_heard():
            self._on_heard()

    def _search(self) -> Optional[ScoutDevice]:
        """Listen on every other device until one hears the user, or the search is over."""
        devices = scout_devices(self._mic.devices.selectable_input_devices(), self._mic.device_name)
        logging.info("Default microphone silent so far; listening on: %s", [device.label for device in devices])
        listeners = self._open_listeners(devices)
        try:
            deadline = time.monotonic() + LISTEN_WINDOW_S
            while time.monotonic() < deadline and not self._current_heard() and not self._stop.wait(_POLL_S):
                for listener in listeners:
                    if self._hears_user(listener):
                        return listener.device
        finally:
            for listener in listeners:
                self._close(listener)
        return None

    def _open_listeners(self, devices: List[ScoutDevice]) -> List[_Listener]:
        """Open the devices that allow it; one that refuses is simply left out."""
        listeners = []
        for device in devices:
            candidate = {"index": device.index, "name": device.name, "channels": 1,
                         "samplerate": self._mic.target_samplerate}
            try:
                stream = open_input_stream(self._mic.pyaudio(), candidate, self._mic.chunk_size)
            except Exception as e:
                logging.info("Could not listen on %r: %s", device.label, e)
                continue
            listeners.append(_Listener(device, stream))
        return listeners

    def _hears_user(self, listener: _Listener) -> bool:
        """Read what the device has so far; True once it clearly beats the microphone in use."""
        try:
            while listener.stream.get_read_available() >= self._mic.chunk_size:
                pcm = listener.stream.read(self._mic.chunk_size, exception_on_overflow=False)
                level = _peak_dbfs(pcm)
                if level >= SIGNAL_DBFS and level - self._current_peak_db >= SWITCH_MARGIN_DB:
                    listener.loud_blocks += 1
        except Exception as e:
            logging.info("Stopped listening on %r: %s", listener.device.label, e)
            return False
        return listener.loud_blocks >= MIN_SIGNAL_BLOCKS

    @staticmethod
    def _close(listener: _Listener) -> None:
        """Release a device; it may already be gone."""
        try:
            listener.stream.close()
        except Exception as e:
            logging.debug("Closing %r failed: %s", listener.device.label, e)
