"""Per-model capabilities from the curated catalog decide the request fields."""
from __future__ import annotations

import pytest

from app.core.models import (
    REPHRASING_MODEL_OPTIONS,
    REPHRASING_MODELS,
    prompt_token_limit,
    rephrasing_supports_temperature,
    transcription_form_fields,
)


@pytest.mark.parametrize("model", ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5.5", "gpt-5.6-luna", "gpt-6-luna"])
def test_reasoning_models_keep_their_default_temperature(model):
    assert not rephrasing_supports_temperature(model)


@pytest.mark.parametrize("model", ["gpt-5.2", "gpt-5.4-mini", "gpt-4.1-mini", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"])
def test_other_models_accept_a_temperature(model):
    assert rephrasing_supports_temperature(model)


def test_dated_snapshot_and_display_annotation_share_the_catalog_entry():
    assert not rephrasing_supports_temperature("gpt-5-mini-2025-08-07")
    assert not rephrasing_supports_temperature("GPT-5-Mini (openai)")


def test_unlisted_model_is_treated_as_a_plain_compatible_model():
    assert rephrasing_supports_temperature("my-local-llm")
    assert transcription_form_fields("my-whisper", "Hello", 0.2, "de") == {
        "model": "my-whisper", "prompt": "Hello", "temperature": 0.2, "language": "de"}


def test_transcription_fields_follow_the_model_capabilities():
    assert transcription_form_fields("gpt-4o-transcribe-diarize", "Hello", 0.2, "de") == {
        "model": "gpt-4o-transcribe-diarize", "response_format": "json", "chunking_strategy": "auto", "language": "de"}
    assert transcription_form_fields("gpt-transcribe", "Hello", 0.2, "de") == {
        "model": "gpt-transcribe", "prompt": "Hello", "languages[]": "de"}
    assert prompt_token_limit("whisper-large-v3") == 230
    assert prompt_token_limit("gpt-4o-transcribe") is None


def test_dropdown_lists_come_from_the_same_catalog():
    assert sum(len(models) for models in REPHRASING_MODEL_OPTIONS.values()) == len(REPHRASING_MODELS)
    assert REPHRASING_MODEL_OPTIONS["groq"][0] == "openai/gpt-oss-120b"


def test_a_model_takes_a_temperature_only_when_it_does_not_reason_by_default():
    for model_id, spec in REPHRASING_MODELS.items():
        if spec.provider == "openai" and spec.reasoning_efforts:
            assert spec.temperature == (spec.default_reasoning == "none"), model_id

