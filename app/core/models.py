"""Model catalogs and per-model API capabilities — the single source for requests and settings UI."""
from __future__ import annotations

from typing import Any, Dict

from app.core.textutil import clean_model_name

# Curated provider catalogs, verified against official documentation on 2026-10-09.
TRANSCRIPTION_MODEL_OPTIONS = {
    "openai": ["gpt-transcribe", "gpt-4o-transcribe", "gpt-4o-mini-transcribe",
               "gpt-4o-mini-transcribe-2025-12-15", "gpt-4o-transcribe-diarize", "whisper-1"],
    "groq": ["whisper-large-v3-turbo", "whisper-large-v3"],
}
REPHRASING_MODEL_OPTIONS = {
    "openai": ["gpt-6-luna", "gpt-6.1-sol", "gpt-6-astra", "gpt-6-sol", "gpt-5.6-luna",
               "gpt-5.6-terra", "gpt-5.6-sol", "gpt-4.1-mini", "gpt-4.1", "gpt-4.1-nano", "gpt-4o-mini", "gpt-4o"],
    "groq": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
}
DEFAULT_TRANSCRIPTION_MODEL = "whisper-1"
DEFAULT_REPHRASING_MODEL = "gpt-6-luna"

# GPT-5.6/GPT-6 reasoning chat models accept only their default temperature.
_FIXED_TEMPERATURE_REPHRASING_FAMILIES = ("gpt-5.6", "gpt-6")
_GPT_TRANSCRIBE = "gpt-transcribe"
_DIARIZE = "gpt-4o-transcribe-diarize"


def rephrasing_supports_temperature(model: str) -> bool:
    """False for chat models that reject a configurable temperature (HTTP 400 otherwise)."""
    return not clean_model_name(model).lower().startswith(_FIXED_TEMPERATURE_REPHRASING_FAMILIES)


def transcription_supports_temperature(model: str) -> bool:
    """False for transcription models that reject the ``temperature`` field."""
    return clean_model_name(model) not in (_GPT_TRANSCRIBE, _DIARIZE)


def transcription_supports_prompt(model: str) -> bool:
    """False for the diarization model, which takes no prompt."""
    return clean_model_name(model) != _DIARIZE


def transcription_form_fields(model: str, prompt: str, temperature: float, language: str) -> Dict[str, Any]:
    """Build the multipart form fields the selected transcription model accepts."""
    model = clean_model_name(model)
    data: Dict[str, Any] = {"model": model}
    if transcription_supports_prompt(model):
        data["prompt"] = prompt
    else:
        data.update(response_format="json", chunking_strategy="auto")
    if transcription_supports_temperature(model):
        data["temperature"] = temperature
    # An empty language lets the API auto-detect.
    if language:
        data["languages[]" if model == _GPT_TRANSCRIBE else "language"] = language
    return data
