"""Tests for app/audio/store.py (RecordingStore)."""
from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import pytest

from app.audio.store import RecordingStore


@pytest.fixture
def store(tmp_path: Path) -> RecordingStore:
    return RecordingStore(str(tmp_path))


def create_recording(store: RecordingStore, when: datetime, mtime: float) -> str:
    path = store.new_path(when)
    with open(path, "wb") as f:
        f.write(b"RIFF")
    os.utime(path, (mtime, mtime))
    return path


class TestRecordingStore:
    def test_new_path_uses_prefix_suffix_and_directory(self, store: RecordingStore):
        path = store.new_path(datetime(2026, 7, 11, 12, 30, 45))
        name = os.path.basename(path)
        assert name == "whispertyper_recording_20260711_123045.wav"
        assert os.path.dirname(path) == store.directory

    def test_empty_directory(self, store: RecordingStore):
        assert store.list() == []
        assert store.latest() is None
        assert store.exists() is False

    def test_list_is_sorted_newest_first_by_mtime(self, store: RecordingStore):
        now = time.time()
        old = create_recording(store, datetime(2026, 1, 1, 10, 0, 0), now - 100)
        new = create_recording(store, datetime(2026, 1, 1, 9, 0, 0), now)  # older name, newer mtime
        assert store.list() == [new, old]
        assert store.latest() == new

    def test_unrelated_files_are_ignored(self, store: RecordingStore, tmp_path: Path):
        (tmp_path / "other.wav").write_bytes(b"x")
        (tmp_path / "whispertyper_recording_x.txt").write_bytes(b"x")
        assert store.list() == []

    def test_keep_only_latest(self, store: RecordingStore):
        now = time.time()
        create_recording(store, datetime(2026, 1, 1, 10, 0, 0), now - 100)
        newest = create_recording(store, datetime(2026, 1, 1, 11, 0, 0), now)
        store.keep_only_latest()
        assert store.list() == [newest]

    def test_cleanup_all(self, store: RecordingStore):
        now = time.time()
        create_recording(store, datetime(2026, 1, 1, 10, 0, 0), now - 100)
        create_recording(store, datetime(2026, 1, 1, 11, 0, 0), now)
        store.cleanup_all()
        assert store.list() == []
        assert store.exists() is False
