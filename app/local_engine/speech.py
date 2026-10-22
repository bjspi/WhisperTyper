"""Audio handling of the engine process: decoding, silence removal and the hallucination filter."""
from __future__ import annotations

import io
from typing import Any, List

SAMPLE_RATE = 16000
#: Speech shorter than this after silence removal is not transcribed (Whisper invents text for it).
MIN_SPEECH_SAMPLES = SAMPLE_RATE // 4

# Whole-text outputs Whisper is known to invent for (near-)silent audio, per language.
HALLUCINATIONS = {
    "de": {
        "vielen dank.", "vielen dank fürs zuschauen.", "vielen dank für's zuschauen!", "danke.",
        "untertitel im auftrag des zdf, 2017", "untertitel im auftrag des zdf, 2018",
        "untertitel im auftrag des zdf, 2020", "untertitel der amara.org-community",
        "untertitelung aufgrund der amara.org-community", "copyright wdr 2021", "swr 2020", "tschüss.",
        "bis zum nächsten mal.",
    },
    "en": {"thank you.", "thanks for watching!", "thank you for watching.", "you", "bye.", "bye!"},
    "fr": {"merci.", "sous-titres réalisés par la communauté d'amara.org"},
    "es": {"gracias.", "¡gracias por ver!", "subtítulos realizados por la comunidad de amara.org"},
}


def decode_audio(data: bytes) -> Any:
    """Decode any container PyAV/FFmpeg reads into 16 kHz mono float32 samples."""
    import av
    import numpy as np

    chunks: List[Any] = []
    with av.open(io.BytesIO(data), mode="r") as container:
        if not container.streams.audio:
            raise ValueError("no audio stream")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
        for frame in container.decode(stream):
            chunks.extend(out.to_ndarray().reshape(-1) for out in resampler.resample(frame))
        chunks.extend(out.to_ndarray().reshape(-1) for out in resampler.resample(None))
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)


def strip_silence(audio: Any) -> Any:
    """Keep only the speech (Silero VAD); pauses cost time and make Whisper invent text."""
    import numpy as np
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    chunks = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=500, speech_pad_ms=300))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate([audio[chunk["start"]:chunk["end"]] for chunk in chunks])


def is_hallucination(text: str, language: str) -> bool:
    """Whether ``text`` is a known silence hallucination (all languages when none is given)."""
    normalized = text.strip().lower()
    if language:
        return normalized in HALLUCINATIONS.get(language, set())
    return any(normalized in phrases for phrases in HALLUCINATIONS.values())
