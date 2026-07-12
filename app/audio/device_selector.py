"""Microphone input-device enumeration and stream-candidate prioritization.

Single responsibility: turn the platform's PyAudio device list into (a) the names shown in the
settings dropdown and (b) an ordered list of open-stream candidates for the capture code to try.
Pulled out of AudioMixin to keep recording lifecycle and device discovery separate.

The owning app supplies the shared PyAudio instance, the config dict and the app sample rate;
this class holds no audio state of its own.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

import pyaudio

from app.core.env import is_MACOS
from app.core.textutil import demojibake


class InputDeviceSelector:
    """Enumerates input devices and builds prioritized open-stream candidates."""

    def __init__(self, owner: Any) -> None:
        """
        Args:
            owner: The app; provides ``config``, ``samplerate`` and
                ``_get_input_pyaudio_instance()``.
        """
        self._owner = owner

    @property
    def _config(self) -> Dict[str, Any]:
        return self._owner.config

    @property
    def _samplerate(self) -> int:
        return self._owner.samplerate

    def selectable_input_devices(self) -> List[Dict[str, Any]]:
        """Input devices on the platform's default host API, for the settings dropdown."""
        devices: List[Dict[str, Any]] = []
        # Timing instrumentation: PyAudio enumeration can be slow (esp. the first call after
        # start), which blocks the settings window from showing. Log where the time goes.
        t0 = time.perf_counter()
        try:
            audio = self._owner._get_input_pyaudio_instance()
            t_instance = time.perf_counter()
            default_host = int(audio.get_default_host_api_info().get("index", 0))
            device_count = audio.get_device_count()
            t_meta = time.perf_counter()
        except Exception as e:
            logging.warning(f"Could not enumerate input devices: {e}")
            return devices

        seen: set = set()
        for index in range(device_count):
            try:
                info = audio.get_device_info_by_index(index)
            except Exception:
                continue
            if int(info.get("maxInputChannels", 0) or 0) <= 0:
                continue
            if int(info.get("hostApi", -1)) != default_host:
                continue
            # PyAudio/MME returns names in a mangled encoding (e.g. 'IntelÂ®'); repair for
            # display + storage so it matches on the next enumeration.
            name = demojibake(str(info.get("name", f"Input {index}")))
            # The "Sound Mapper" pseudo-device is just the system default (already offered separately).
            if "mapper" in name.lower() or name in seen:
                continue
            seen.add(name)
            devices.append({"index": int(info.get("index", index)), "name": name})
        t_end = time.perf_counter()
        logging.info(
            "Input device enumeration: %d device(s) from %d entries in %.0fms "
            "(pyaudio_instance=%.0fms, host/count=%.0fms, per-device loop=%.0fms)",
            len(devices), device_count, (t_end - t0) * 1000,
            (t_instance - t0) * 1000, (t_meta - t_instance) * 1000, (t_end - t_meta) * 1000,
        )
        return devices

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
            match = next((d for d in self.selectable_input_devices() if d["name"] == configured), None)
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
                logging.warning(f"Configured input device {configured!r} not found; using system default.")
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
