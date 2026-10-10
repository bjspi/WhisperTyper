"""Curated model catalogs with per-model API capabilities — the single source for requests and settings UI.

Each model is listed once with what it accepts; the per-provider dropdown lists are derived from
the same tables. Models that are not listed (typed by hand, custom endpoints) get the defaults of
:class:`ModelSpec`, which describe a plain OpenAI-compatible model.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from app.core.textutil import clean_model_name

#: Approximate maximum token length of the Whisper initial prompt.
WHISPER_PROMPT_TOKEN_LIMIT = 230


class ModelSpec(NamedTuple):
    """What a model accepts in a request."""

    provider: str
    #: Accepts a configurable temperature; reasoning models reject anything but their default (HTTP 400).
    temperature: bool = True
    #: Transcription: token budget of the initial prompt, or None without a known limit.
    prompt_token_limit: Optional[int] = None
    #: Transcription: speaker diarization — takes no prompt and requires ``chunking_strategy``.
    diarization: bool = False
    #: Transcription: the language is sent as ``languages[]`` instead of ``language``.
    language_list: bool = False
    #: Rephrasing: accepted ``reasoning_effort`` values, empty for models without reasoning control.
    reasoning_efforts: Tuple[str, ...] = ()
    #: Rephrasing: the effort the model uses when none is sent (None where the provider does not document it).
    default_reasoning: Optional[str] = None


# Verified against the providers' official model pages on 2026-10-10 (active models only).
TRANSCRIPTION_MODELS: Dict[str, ModelSpec] = {
    "gpt-transcribe": ModelSpec("openai", temperature=False, language_list=True),
    "gpt-4o-transcribe": ModelSpec("openai"),
    "gpt-4o-mini-transcribe": ModelSpec("openai"),
    "gpt-4o-mini-transcribe-2025-12-15": ModelSpec("openai"),
    "gpt-4o-transcribe-diarize": ModelSpec("openai", temperature=False, diarization=True),
    "whisper-1": ModelSpec("openai", prompt_token_limit=WHISPER_PROMPT_TOKEN_LIMIT),
    "whisper-large-v3-turbo": ModelSpec("groq", prompt_token_limit=WHISPER_PROMPT_TOKEN_LIMIT),
    "whisper-large-v3": ModelSpec("groq", prompt_token_limit=WHISPER_PROMPT_TOKEN_LIMIT),
}

# Reasoning effort scales as documented per model family.
_GPT5_EFFORTS = ("minimal", "low", "medium", "high")
_GPT5X_EFFORTS = ("none", "low", "medium", "high", "xhigh")
_GPT56_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")
_GPT6_HIGH_EFFORTS = ("low", "medium", "high", "xhigh", "max")
_GPT_OSS_EFFORTS = ("low", "medium", "high")

# A model only accepts a temperature while it does not reason: true for models whose default
# effort is "none"; models that reason by default get no temperature (HTTP 400 otherwise).
REPHRASING_MODELS: Dict[str, ModelSpec] = {
    "gpt-6-luna": ModelSpec("openai", temperature=False, reasoning_efforts=_GPT56_EFFORTS, default_reasoning="medium"),
    "gpt-6.1-sol": ModelSpec("openai", temperature=False, reasoning_efforts=_GPT6_HIGH_EFFORTS,
                             default_reasoning="medium"),
    "gpt-6-astra": ModelSpec("openai", temperature=False, reasoning_efforts=_GPT6_HIGH_EFFORTS),
    "gpt-6-sol": ModelSpec("openai", temperature=False, reasoning_efforts=_GPT56_EFFORTS, default_reasoning="medium"),
    "gpt-5.6-luna": ModelSpec("openai", temperature=False, reasoning_efforts=_GPT56_EFFORTS,
                              default_reasoning="medium"),
    "gpt-5.6-terra": ModelSpec("openai", temperature=False, reasoning_efforts=_GPT56_EFFORTS,
                               default_reasoning="medium"),
    "gpt-5.6-sol": ModelSpec("openai", temperature=False, reasoning_efforts=_GPT56_EFFORTS,
                             default_reasoning="medium"),
    "gpt-5.5": ModelSpec("openai", temperature=False, reasoning_efforts=_GPT5X_EFFORTS, default_reasoning="medium"),
    "gpt-5.4": ModelSpec("openai", reasoning_efforts=_GPT5X_EFFORTS, default_reasoning="none"),
    "gpt-5.4-mini": ModelSpec("openai", reasoning_efforts=_GPT5X_EFFORTS, default_reasoning="none"),
    "gpt-5.2": ModelSpec("openai", reasoning_efforts=_GPT5X_EFFORTS, default_reasoning="none"),
    "gpt-5": ModelSpec("openai", temperature=False, reasoning_efforts=_GPT5_EFFORTS),
    "gpt-5-mini": ModelSpec("openai", temperature=False, reasoning_efforts=_GPT5_EFFORTS),
    "gpt-5-nano": ModelSpec("openai", temperature=False, reasoning_efforts=_GPT5_EFFORTS),
    "gpt-4.1-mini": ModelSpec("openai"),
    "gpt-4.1": ModelSpec("openai"),
    "gpt-4.1-nano": ModelSpec("openai"),
    "gpt-4o-mini": ModelSpec("openai"),
    "gpt-4o": ModelSpec("openai"),
    "openai/gpt-oss-120b": ModelSpec("groq", reasoning_efforts=_GPT_OSS_EFFORTS),
    "openai/gpt-oss-20b": ModelSpec("groq", reasoning_efforts=_GPT_OSS_EFFORTS),
    "llama-3.3-70b-versatile": ModelSpec("groq"),
    "llama-3.1-8b-instant": ModelSpec("groq"),
}

DEFAULT_TRANSCRIPTION_MODEL = "whisper-1"
DEFAULT_REPHRASING_MODEL = "gpt-6-luna"

_UNLISTED = ModelSpec("custom")
# OpenAI's pinned snapshots append a release date to the model ID ("gpt-5-mini-2025-08-07").
_SNAPSHOT_DATE = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def _options_by_provider(catalog: Dict[str, ModelSpec]) -> Dict[str, List[str]]:
    """Dropdown entries per provider, in catalog order."""
    options: Dict[str, List[str]] = {}
    for model_id, spec in catalog.items():
        options.setdefault(spec.provider, []).append(model_id)
    return options


TRANSCRIPTION_MODEL_OPTIONS = _options_by_provider(TRANSCRIPTION_MODELS)
REPHRASING_MODEL_OPTIONS = _options_by_provider(REPHRASING_MODELS)


def _spec(catalog: Dict[str, ModelSpec], model: str) -> ModelSpec:
    """Capabilities of ``model``; a dated snapshot shares its base model's entry."""
    model_id = clean_model_name(model).lower()
    return catalog.get(model_id) or catalog.get(_SNAPSHOT_DATE.sub("", model_id)) or _UNLISTED


def rephrasing_supports_temperature(model: str) -> bool:
    """False for chat models that reject a configurable temperature."""
    return _spec(REPHRASING_MODELS, model).temperature


def transcription_supports_temperature(model: str) -> bool:
    """False for transcription models that reject the ``temperature`` field."""
    return _spec(TRANSCRIPTION_MODELS, model).temperature


def transcription_supports_prompt(model: str) -> bool:
    """False for the diarization model, which takes no prompt."""
    return not _spec(TRANSCRIPTION_MODELS, model).diarization


def prompt_token_limit(model: str) -> int | None:
    """Prompt token budget of a transcription model, or None when it has no known limit."""
    return _spec(TRANSCRIPTION_MODELS, model).prompt_token_limit


def transcription_form_fields(model: str, prompt: str, temperature: float, language: str) -> Dict[str, Any]:
    """Build the multipart form fields the selected transcription model accepts."""
    model = clean_model_name(model)
    spec = _spec(TRANSCRIPTION_MODELS, model)
    data: Dict[str, Any] = {"model": model}
    if spec.diarization:
        data.update(response_format="json", chunking_strategy="auto")
    else:
        data["prompt"] = prompt
    if spec.temperature:
        data["temperature"] = temperature
    # An empty language lets the API auto-detect.
    if language:
        data["languages[]" if spec.language_list else "language"] = language
    return data
