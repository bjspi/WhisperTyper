"""RecordingStore — the single source of truth for on-disk recordings.

All ``whispertyper_recording_*.wav`` listing goes through this class, with one
consistent ordering (newest by modification time first).

Single responsibility: locate / enumerate / prune recording files. No Qt, no audio.
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime
from typing import List, Optional


class RecordingStore:
    """Locates, lists and prunes WhisperTyper recording files in a directory."""

    PREFIX = "whispertyper_recording_"
    SUFFIX = ".wav"

    def __init__(self, directory: Optional[str] = None) -> None:
        """Bind the store to a directory (defaults to the system temp dir)."""
        self._dir = directory or tempfile.gettempdir()

    @property
    def directory(self) -> str:
        """Directory the recordings live in."""
        return self._dir

    def new_path(self, when: Optional[datetime] = None) -> str:
        """Build a path for a new recording, timestamped to the second."""
        stamp = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
        return os.path.join(self._dir, f"{self.PREFIX}{stamp}{self.SUFFIX}")

    def list(self) -> List[str]:
        """Return all recording paths, newest (by mtime) first."""
        try:
            names = [
                n for n in os.listdir(self._dir)
                if n.startswith(self.PREFIX) and n.endswith(self.SUFFIX)
            ]
        except OSError:
            return []
        paths = [os.path.join(self._dir, n) for n in names]
        paths.sort(key=self._mtime, reverse=True)
        return paths

    def latest(self) -> Optional[str]:
        """Return the newest recording path, or None if there is none."""
        items = self.list()
        return items[0] if items else None

    def exists(self) -> bool:
        """True if at least one recording is present."""
        return self.latest() is not None

    def keep_only_latest(self) -> None:
        """Delete every recording except the newest one."""
        for path in self.list()[1:]:
            self._remove(path)

    def cleanup_all(self) -> None:
        """Delete all recordings (used on startup)."""
        for path in self.list():
            self._remove(path)

    @staticmethod
    def _mtime(path: str) -> float:
        try:
            return os.path.getmtime(path)
        except OSError:
            return 0.0

    @staticmethod
    def _remove(path: str) -> None:
        try:
            os.remove(path)
        except OSError as e:
            logging.warning(f"Could not delete old recording {os.path.basename(path)}: {e}")
