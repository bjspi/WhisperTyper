"""What the first-start guide does to the configuration: the same rules as the settings, in one step."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.core.api_keys import LOCAL_ENDPOINT, PROVIDER_NAMES, provider_endpoint
from app.core.models import (
    DEFAULT_REPHRASING_MODEL,
    DEFAULT_TRANSCRIPTION_MODEL,
    REPHRASING_MODEL_OPTIONS,
    TRANSCRIPTION_MODEL_OPTIONS,
)
from app.core.prompts import default_transcription_prompt, is_default_transcription_prompt, localize_unedited_entry

#: The provider whose models the app's default model names already are.
_DEFAULT_MODELS_PROVIDER = "openai"


@dataclass(frozen=True)
class ApiAccess:
    """An official provider and the key to use with it; the key may still be missing."""

    provider: str
    key: str = ""


@dataclass(frozen=True)
class OnboardingChoices:
    """The answers of the guide."""

    ui_language: str
    input_language: str
    #: Where transcription runs: None is the local engine.
    transcription: Optional[ApiAccess]
    #: Where rephrasing runs: None leaves it switched off.
    rephrasing: Optional[ApiAccess]


def apply_languages(config: Dict[str, Any], ui_language: str, input_language: str) -> None:
    """Set both languages; prompts the user never edited follow them, as they do in the settings."""
    config["ui_language"] = ui_language
    config["input_language"] = input_language
    if is_default_transcription_prompt(config.get("prompt", "")):
        config["prompt"] = default_transcription_prompt(input_language, ui_language)
    for entry in config.get("post_rephrasing_entries", []):
        localize_unedited_entry(entry, ui_language)


def apply_onboarding(config: Dict[str, Any], choices: OnboardingChoices) -> None:
    """Store every answer of a finished guide."""
    apply_languages(config, choices.ui_language, choices.input_language)
    if choices.transcription is None:
        config["api_endpoint"] = LOCAL_ENDPOINT
        config["local_engine_enabled"] = True
        config["local_engine_setup_pending"] = True
    else:
        provider = choices.transcription.provider
        config["api_endpoint"] = provider_endpoint(provider, "transcription")
        config["model"] = (DEFAULT_TRANSCRIPTION_MODEL if provider == _DEFAULT_MODELS_PROVIDER
                           else TRANSCRIPTION_MODEL_OPTIONS[provider][0])
        _add_key(config, choices.transcription)
    config["rephrasing_enabled"] = choices.rephrasing is not None
    if choices.rephrasing is not None:
        provider = choices.rephrasing.provider
        config["rephrasing_api_url"] = provider_endpoint(provider, "rephrasing")
        config["rephrasing_model"] = (DEFAULT_REPHRASING_MODEL if provider == _DEFAULT_MODELS_PROVIDER
                                      else REPHRASING_MODEL_OPTIONS[provider][0])
        _add_key(config, choices.rephrasing)


def _add_key(config: Dict[str, Any], access: ApiAccess) -> None:
    """Add the key as a profile, unless it is empty or already there."""
    key = access.key.strip()
    profiles = config.setdefault("api_key_profiles", [])
    if not key or any(profile["provider"] == access.provider and profile["key"] == key for profile in profiles):
        return
    profiles.append({"id": uuid.uuid4().hex, "name": PROVIDER_NAMES[access.provider], "provider": access.provider,
                     "key": key})
