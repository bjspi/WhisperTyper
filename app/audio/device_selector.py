"""Microphone input-device enumeration and stream-candidate prioritization.

Single responsibility: turn the platform's PyAudio device list into (a) the names shown in the
settings dropdown and (b) an ordered list of open-stream candidates for the capture code to try.

Callers supply the config accessor, the app sample rate and the shared PyAudio instance;
this class holds no audio state of its own.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Mapping

import pyaudio

from app.core.env import is_MACOS
from app.core.textutil import demojibake

#: MME cuts device names to this many characters; shorter names are complete already.
_MME_NAME_LIMIT = 31


def _longest_completion(name: str, candidates: List[str]) -> str:
    """The longest of ``candidates`` beginning with ``name`` when ``name`` looks cut by MME, else ``name``."""
    if len(name) < _MME_NAME_LIMIT - 1:  # a trailing space may already be stripped
        return name
    longer = [candidate for candidate in candidates if candidate.startswith(name) and len(candidate) > len(name)]
    return max(longer, key=len) if longer else name


class InputDeviceSelector:
    """Enumerates input devices and builds prioritized open-stream candidates."""

    def __init__(self, config: Callable[[], Mapping[str, Any]], samplerate: int,
                 audio: Callable[[], pyaudio.PyAudio]) -> None:
        """
        Args:
            config: Returns the current app config (read for ``input_device_name``).
            samplerate: The app's target capture rate.
            audio: Returns the shared capture PyAudio instance.
        """
        self._get_config = config
        self._samplerate = samplerate
        self._audio = audio
        #: Full names for the names MME cuts at 31 characters, from the other host APIs (display only).
        self._display_names: Dict[str, str] = {}

    @property
    def _config(self) -> Mapping[str, Any]:
        return self._get_config()

    def selectable_input_devices(self) -> List[Dict[str, Any]]:
        """Input devices on the platform's default host API, for the settings dropdown."""
        devices: List[Dict[str, Any]] = []
        # Timing instrumentation: PyAudio enumeration can be slow (esp. the first call after
        # start), which blocks the settings window from showing. Log where the time goes.
        t0 = time.perf_counter()
        try:
            audio = self._audio()
            t_instance = time.perf_counter()
            default_host = int(audio.get_default_host_api_info().get("index", 0))
            device_count = audio.get_device_count()
            t_meta = time.perf_counter()
        except Exception as e:
            logging.warning(f"Could not enumerate input devices: {e}")
            return devices

        seen: set = set()
        other_names: List[str] = []  # input device names of the other host APIs, at full length
        for index in range(device_count):
            try:
                info = audio.get_device_info_by_index(index)
            except Exception:
                continue
            if int(info.get("maxInputChannels", 0) or 0) <= 0:
                continue
            # PyAudio/MME returns names in a mangled encoding (e.g. 'IntelÂ®'); repair for
            # display + storage so it matches on the next enumeration. MME also cuts names
            # at 31 characters, often leaving a trailing space, which the stored name has not.
            name = demojibake(str(info.get("name", f"Input {index}"))).strip()
            if int(info.get("hostApi", -1)) != default_host:
                other_names.append(name)
                continue
            # The "Sound Mapper" pseudo-device is just the system default (already offered separately).
            if "mapper" in name.lower() or name in seen:
                continue
            seen.add(name)
            devices.append({"index": int(info.get("index", index)), "name": name})
        # The stored name stays the one of the default host API; the label is the full name for showing.
        self._display_names = {device["name"]: _longest_completion(device["name"], other_names) for device in devices}
        for device in devices:
            device["label"] = self._display_names[device["name"]]
        t_end = time.perf_counter()
        logging.info(
            "Input device enumeration: %d device(s) from %d entries in %.0fms "
            "(pyaudio_instance=%.0fms, host/count=%.0fms, per-device loop=%.0fms)",
            len(devices), device_count, (t_end - t0) * 1000,
            (t_instance - t0) * 1000, (t_meta - t_instance) * 1000, (t_end - t_meta) * 1000,
        )
        return devices

    def display_name(self, name: str) -> str:
        """The full name of a device for showing it (MME cuts names at 31 characters); other names as they are."""
        if not self._display_names:
            self.selectable_input_devices()
        return self._display_names.get(name, name)

    def preferred_input_stream_candidates(self, audio_instance: pyaudio.PyAudio) -> List[Dict[str, Any]]:
        """Build prioritized input stream candidates for the current platform."""
        if not is_MACOS:
            return self._windows_input_candidates(audio_instance)

        # macOS extension: Use an explicit input device and its native sample rate.
        # This avoids CoreAudio/PortAudio combinations that report an open stream
        # but only return silent frames inside the bundled .app.
        candidates_by_index = {
            device["index"]: {
                "index": device["index"],
                "name": device["name"],
                "channels": device["channels"],
                "samplerate": device["default_samplerate"],
            }
            for device in self._list_input_device_candidates(audio_instance)
        }

        prioritized: List[Dict[str, Any]] = []
        try:
            default_info = audio_instance.get_default_input_device_info()
            default_index = int(default_info.get("index"))
            if default_index in candidates_by_index:
                prioritized.append(candidates_by_index.pop(default_index))
            else:
                default_samplerate = int(round(float(default_info.get("defaultSampleRate", self._samplerate) or self._samplerate)))
                prioritized.append({
                    "index": default_index,
                    "name": str(default_info.get("name", "default input")),
                    "channels": 1,
                    "samplerate": max(8000, default_samplerate),
                })
        except Exception as e:
            logging.warning(f"macOS extension: Could not resolve default input device: {e}")

        prioritized.extend(candidates_by_index.values())

        if prioritized:
            readable_candidates = ", ".join(
                f"{candidate['name']}#{candidate['index']}@{candidate['samplerate']}Hz"
                for candidate in prioritized
            )
            logging.info(f"macOS extension: Input device candidates: {readable_candidates}")
        else:
            logging.warning("macOS extension: No explicit input devices reported by PyAudio. Falling back to generic input open.")

        prioritized.append({
            "index": None,
            "name": "generic default input",
            "channels": 1,
            "samplerate": self._samplerate,
        })
        return prioritized

    def _windows_input_candidates(self, audio_instance: pyaudio.PyAudio) -> List[Dict[str, Any]]:
        """Candidates honouring the user-selected input device, with the system default as fallback."""
        configured = str(self._config.get("input_device_name", "") or "").strip()
        candidates: List[Dict[str, Any]] = []
        if configured:
            devices = self.selectable_input_devices()
            match = next((d for d in devices if d["name"] == configured), None)
            if match:
                try:
                    info = audio_instance.get_device_info_by_index(match["index"])
                    native = int(round(float(info.get("defaultSampleRate", self._samplerate) or self._samplerate)))
                except Exception:
                    native = self._samplerate
                # Prefer the app rate (16 kHz); fall back to the device's native rate (resampled on upload).
                candidates.append({"index": match["index"], "name": match["name"], "channels": 1, "samplerate": self._samplerate})
                if native != self._samplerate:
                    candidates.append({"index": match["index"], "name": match["name"], "channels": 1, "samplerate": max(8000, native)})
            else:
                # A warning with the names on offer: a name that never matches shows up here.
                logging.warning(f"Configured input device {configured!r} is not among the input devices "
                                f"{[d['name'] for d in devices]}; using the system default.")
        candidates.append({"index": None, "name": "default input", "channels": 1, "samplerate": self._samplerate})
        return candidates

    def _list_input_device_candidates(self, audio_instance: pyaudio.PyAudio) -> List[Dict[str, Any]]:
        """Collect usable input devices from the current PyAudio backend."""
        candidates: List[Dict[str, Any]] = []
        try:
            device_count = audio_instance.get_device_count()
        except Exception as e:
            logging.warning(f"Could not enumerate audio input devices: {e}")
            return candidates

        for index in range(device_count):
            try:
                info = audio_instance.get_device_info_by_index(index)
            except Exception as e:
                logging.debug(f"Skipping unreadable audio device #{index}: {e}")
                continue

            max_input_channels = int(info.get("maxInputChannels", 0) or 0)
            if max_input_channels <= 0:
                continue

            default_samplerate = int(round(float(info.get("defaultSampleRate", self._samplerate) or self._samplerate)))
            candidates.append({
                "index": int(info.get("index", index)),
                "name": str(info.get("name", f"Input device {index}")),
                "channels": 1,
                "default_samplerate": max(8000, default_samplerate),
                "max_input_channels": max_input_channels,
            })

        return candidates
