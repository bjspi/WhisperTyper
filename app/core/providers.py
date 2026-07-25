"""Known provider metadata and pure provider/config resolution helpers."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Tuple

PROVIDER_IDS = ("openai", "groq", "custom")
FEATURE_IDS = ("transcription", "rephrasing")

PROVIDER_LABELS = {
    "openai": "OpenAI",
    "groq": "Groq",
    "custom": "Custom (OpenAI-compatible)",
}

PROVIDER_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
}

PROVIDER_ENDPOINTS = {
    "openai": {
        "transcription": f"{PROVIDER_BASE_URLS['openai']}/audio/transcriptions",
        "rephrasing": f"{PROVIDER_BASE_URLS['openai']}/chat/completions",
    },
    "groq": {
        "transcription": f"{PROVIDER_BASE_URLS['groq']}/audio/transcriptions",
        "rephrasing": f"{PROVIDER_BASE_URLS['groq']}/chat/completions",
    },
}

FALLBACK_MODELS = {
    "openai": {
        "transcription": ["gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"],
        "rephrasing": ["gpt-5.6-luna", "gpt-5.6", "gpt-5.4", "gpt-4o-mini"],
    },
    "groq": {
        "transcription": ["whisper-large-v3-turbo", "whisper-large-v3"],
        "rephrasing": [
            "openai/gpt-oss-120b",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "openai/gpt-oss-20b",
        ],
    },
}


def infer_provider(endpoint: str) -> str:
    """Infer a known provider from an endpoint, otherwise return ``custom``."""
    value = (endpoint or "").lower()
    if "api.groq.com" in value:
        return "groq"
    if "api.openai.com" in value:
        return "openai"
    return "custom"


def clean_legacy_model_name(model: str) -> str:
    """Strip the old `` (provider)`` display suffix from persisted model names."""
    value = (model or "").strip()
    for suffix in (" (openai)", " (groq)"):
        if value.lower().endswith(suffix):
            return value[:-len(suffix)].strip()
    return value


def resolve_connection(config: Mapping[str, Any], feature: str) -> Tuple[str, str]:
    """Resolve ``(endpoint, api_key)`` for a feature from the central provider config."""
    if feature not in FEATURE_IDS:
        raise ValueError(f"Unknown provider feature: {feature}")
    provider = str(config.get(f"{feature}_provider", "openai"))
    if provider in PROVIDER_ENDPOINTS:
        keys = config.get("provider_api_keys", {})
        api_key = keys.get(provider, "") if isinstance(keys, Mapping) else ""
        return PROVIDER_ENDPOINTS[provider][feature], str(api_key or "")

    custom = config.get("custom_provider_settings", {})
    feature_config = custom.get(feature, {}) if isinstance(custom, Mapping) else {}
    if not isinstance(feature_config, Mapping):
        feature_config = {}
    return str(feature_config.get("endpoint", "") or ""), str(feature_config.get("api_key", "") or "")


def cached_models(config: Mapping[str, Any], provider: str, feature: str) -> List[str]:
    """Return cached models for provider/feature, falling back to maintained defaults."""
    cache = config.get("provider_model_cache", {})
    provider_cache = cache.get(provider, {}) if isinstance(cache, Mapping) else {}
    values = provider_cache.get(feature, []) if isinstance(provider_cache, Mapping) else []
    if not isinstance(values, list) or not values:
        values = FALLBACK_MODELS.get(provider, {}).get(feature, [])
    return unique_models(values)


def unique_models(models: Iterable[Any]) -> List[str]:
    """Normalize, deduplicate and sort model IDs without losing the preferred first item."""
    result: List[str] = []
    seen = set()
    for model in models:
        value = str(model or "").strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def filter_models(payload: Mapping[str, Any], feature: str) -> List[str]:
    """Filter an OpenAI-compatible ``/models`` payload for one WhisperTyper feature."""
    rows = payload.get("data", [])
    if not isinstance(rows, list):
        return []
    selected: List[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        model_id = str(row.get("id", "") or "").strip()
        if not model_id:
            continue
        inputs = {str(x).lower() for x in row.get("input_modalities", []) or []}
        outputs = {str(x).lower() for x in row.get("output_modalities", []) or []}
        lower = model_id.lower()

        if inputs or outputs:
            if feature == "transcription":
                include = "audio" in inputs and bool(outputs & {"text", "transcription"})
            else:
                include = "text" in inputs and "text" in outputs
        elif feature == "transcription":
            include = "whisper" in lower or "transcribe" in lower
        else:
            excluded = (
                "whisper", "transcribe", "embedding", "moderation", "guard", "safeguard",
                "tts", "speech", "realtime", "image", "audio-preview",
            )
            include = not any(token in lower for token in excluded)

        if feature == "rephrasing" and any(token in lower for token in ("guard", "safeguard")):
            include = False
        if include:
            selected.append(model_id)
    return sorted(unique_models(selected), key=str.lower)


def ui_connection(
    provider: str,
    feature: str,
    provider_keys: Mapping[str, str],
    custom_settings: Mapping[str, Mapping[str, str]],
) -> Tuple[str, str]:
    """Resolve current unsaved UI values using the same rules as saved config."""
    temporary: Dict[str, Any] = {
        f"{feature}_provider": provider,
        "provider_api_keys": dict(provider_keys),
        "custom_provider_settings": {
            key: dict(value) for key, value in custom_settings.items()
        },
    }
    return resolve_connection(temporary, feature)
