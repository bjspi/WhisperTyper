"""Tests for app/services/bad_recordings.py."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from app.services import bad_recordings

NOW = datetime(2026, 10, 14, 18, 30, 5)


def make_source(tmp_path: Path) -> Path:
    source = tmp_path / "whispertyper_recording_20261014_183000.wav"
    source.write_bytes(b"RIFFdata")
    return source


def test_copies_audio_and_writes_sidecar(tmp_path: Path):
    source = make_source(tmp_path)
    details = {"form_fields": {"model": "whisper-large-v3-turbo", "prompt": "", "temperature": 0.1},
               "transcription": " kein punkt"}

    saved = bad_recordings.save(str(tmp_path / "bad"), str(source), details, now=NOW)

    assert Path(saved).name == "bad_20261014_183005.wav"
    assert Path(saved).read_bytes() == b"RIFFdata"
    sidecar = json.loads((tmp_path / "bad" / "bad_20261014_183005.json").read_text(encoding="utf-8"))
    assert sidecar["form_fields"]["temperature"] == 0.1
    assert sidecar["transcription"] == " kein punkt"
    assert sidecar["source_file"] == str(source)
    assert source.exists()


def test_same_second_does_not_overwrite(tmp_path: Path):
    source = make_source(tmp_path)
    first = bad_recordings.save(str(tmp_path), str(source), {}, now=NOW)
    second = bad_recordings.save(str(tmp_path), str(source), {}, now=NOW)
    assert first != second
    assert Path(second).name == "bad_20261014_183005_2.wav"


def test_missing_details_are_noted(tmp_path: Path):
    saved = bad_recordings.save(str(tmp_path / "bad"), str(make_source(tmp_path)), None, now=NOW)
    sidecar = json.loads(Path(saved).with_suffix(".json").read_text(encoding="utf-8"))
    assert "note" in sidecar
