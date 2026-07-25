"""Tests for central provider resolution and live model filtering."""
from __future__ import annotations

from app.core.providers import (
    filter_models,
    infer_provider,
    resolve_connection,
)


def test_known_provider_connection_uses_shared_key() -> None:
    config = {
        "transcription_provider": "groq",
        "rephrasing_provider": "groq",
        "provider_api_keys": {"groq": "gsk-one-key"},
    }
    trans_endpoint, trans_key = resolve_connection(config, "transcription")
    chat_endpoint, chat_key = resolve_connection(config, "rephrasing")
    assert trans_endpoint.endswith("/audio/transcriptions")
    assert chat_endpoint.endswith("/chat/completions")
    assert trans_key == chat_key == "gsk-one-key"


def test_custom_connections_remain_feature_specific() -> None:
    config = {
        "transcription_provider": "custom",
        "rephrasing_provider": "custom",
        "custom_provider_settings": {
            "transcription": {"endpoint": "https://stt.example/v1", "api_key": "stt"},
            "rephrasing": {"endpoint": "https://chat.example/v1", "api_key": "chat"},
        },
    }
    assert resolve_connection(config, "transcription") == ("https://stt.example/v1", "stt")
    assert resolve_connection(config, "rephrasing") == ("https://chat.example/v1", "chat")


def test_provider_inference() -> None:
    assert infer_provider("https://api.groq.com/openai/v1/audio/transcriptions") == "groq"
    assert infer_provider("https://api.openai.com/v1/chat/completions") == "openai"
    assert infer_provider("https://local.example/api") == "custom"


def test_model_filter_uses_modalities_and_excludes_guards() -> None:
    payload = {
        "data": [
            {"id": "whisper", "input_modalities": ["audio"], "output_modalities": ["text"]},
            {"id": "chat", "input_modalities": ["text"], "output_modalities": ["text"]},
            {"id": "chat-guard", "input_modalities": ["text"], "output_modalities": ["text"]},
        ]
    }
    assert filter_models(payload, "transcription") == ["whisper"]
    assert filter_models(payload, "rephrasing") == ["chat"]


def test_model_filter_has_openai_compatible_heuristic_fallback() -> None:
    payload = {"data": [{"id": "whisper-1"}, {"id": "gpt-5"}, {"id": "text-embedding-3"}]}
    assert filter_models(payload, "transcription") == ["whisper-1"]
    assert filter_models(payload, "rephrasing") == ["gpt-5"]
