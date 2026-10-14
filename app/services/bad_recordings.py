"""Keep a recording the user flagged as badly transcribed, together with the request behind it.

Each flagged recording becomes ``bad_<timestamp>.<ext>`` plus a ``.json`` sidecar with the sent
form fields (model, prompt, temperature, language), the endpoint and the returned transcript, so
the case can be replayed against other settings later. No Qt here.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, Optional


def save(dest_dir: str, audio_path: str, details: Optional[Dict[str, Any]], *,
         now: Optional[datetime] = None) -> str:
    """Copy ``audio_path`` into ``dest_dir`` with a JSON sidecar; returns the copied audio path.

    Raises:
        OSError: When the directory cannot be created or a file cannot be written.
    """
    os.makedirs(dest_dir, exist_ok=True)
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    ext = os.path.splitext(audio_path)[1] or ".wav"
    stem = os.path.join(dest_dir, f"bad_{stamp}")
    suffix = 1
    while os.path.exists(stem + ext) or os.path.exists(stem + ".json"):
        suffix += 1
        stem = os.path.join(dest_dir, f"bad_{stamp}_{suffix}")
    target = stem + ext
    shutil.copy2(audio_path, target)
    sidecar = {
        "audio_file": os.path.basename(target),
        "source_file": audio_path,
        "flagged_at": (now or datetime.now()).isoformat(timespec="seconds"),
        **(details or {"note": "no transcription request recorded for this file"}),
    }
    with open(stem + ".json", "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, ensure_ascii=False, indent=2)
    return target
