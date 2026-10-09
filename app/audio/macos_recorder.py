"""Native macOS microphone recording through AVAudioRecorder (16-bit mono PCM WAV)."""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

from app.core.env import is_MACOS
from app.platform.macos import NSURL, AVAudioRecorder

SAMPLERATE = 44100
DEVICE_NAME = "macOS system microphone"
#: Give AVAudioRecorder a moment to finalize the WAV header after stop().
_FINALIZE_WAIT_S = 0.08


def available() -> bool:
    """Whether the AVFoundation recorder can be used on this system."""
    return bool(is_MACOS and AVAudioRecorder and NSURL)


class MacRecorder:
    """One AVAudioRecorder session at a time, writing straight to the recording path."""

    def __init__(self) -> None:
        """Start idle."""
        self._recorder: Any = None
        self._path: Optional[str] = None

    @property
    def active(self) -> bool:
        """Whether a recording is in progress."""
        return self._recorder is not None

    def start(self, path: str) -> None:
        """Record to ``path``; raises if AVFoundation is unavailable or refuses to start."""
        if not is_MACOS or AVAudioRecorder is None or NSURL is None:
            raise RuntimeError("Native macOS audio recorder is not available.")
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
        settings = {
            "AVFormatIDKey": int.from_bytes(b"lpcm", "big"),
            "AVSampleRateKey": float(SAMPLERATE),
            "AVNumberOfChannelsKey": 1,
            "AVLinearPCMBitDepthKey": 16,
            "AVLinearPCMIsBigEndianKey": False,
            "AVLinearPCMIsFloatKey": False,
        }
        recorder = AVAudioRecorder.alloc().initWithURL_settings_error_(NSURL.fileURLWithPath_(path), settings, None)
        recorder.setMeteringEnabled_(True)
        recorder.prepareToRecord()
        if not recorder.record():
            raise RuntimeError("AVAudioRecorder did not start recording.")
        self._recorder, self._path = recorder, path
        logging.info(f"macOS: started native recorder capture to '{path}' at {SAMPLERATE} Hz.")

    def stop(self, discard: bool = False) -> Optional[str]:
        """Stop recording; return the WAV path, or None if nothing was recorded or it was discarded."""
        recorder, path = self._recorder, self._path
        self._recorder, self._path = None, None
        if recorder:
            try:
                recorder.stop()
            except Exception as e:
                logging.warning(f"macOS: failed to stop native recorder cleanly: {e}")
        if not path:
            return None
        time.sleep(_FINALIZE_WAIT_S)
        if not discard:
            return path
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                logging.warning(f"macOS: could not delete discarded recording '{path}': {e}")
        return None

    def level(self) -> Optional[float]:
        """Current input level 0..1 from the recorder's meters, or None when unavailable."""
        if not self._recorder:
            return None
        try:
            self._recorder.updateMeters()
            average_power = float(self._recorder.averagePowerForChannel_(0))
        except Exception:
            return None
        return max(0.0, min(1.0, (average_power + 60.0) / 60.0))
