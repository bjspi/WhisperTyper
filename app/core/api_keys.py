"""Provider-scoped API key profiles and migration of the former inline credentials."""
from __future__ import annotations

import uuid
from typing import Any, Mapping
from urllib.parse import urlsplit

TASK_KEY_FIELDS = {
    "transcription": ("api_endpoint", "transcription_key_profile_id", "api_key"),
    "rephrasing": ("rephrasing_api_url", "rephrasing_key_profile_id", "rephrasing_api_key"),
}
PROVIDER_NAMES = {"openai": "OpenAI", "groq": "Groq", "custom": "Custom"}


def provider_for_url(url: str) -> str:
    """Recognize official provider hosts, without accepting lookalike domains."""
    try:
        host = urlsplit(url.strip()).hostname
    except ValueError:
        return "custom"
    return {"api.openai.com": "openai", "api.groq.com": "groq"}.get(host or "", "custom")


def selected_api_key(config: Mapping[str, Any], task: str) -> str:
    """Resolve the explicitly selected profile only when it matches the task's provider."""
    endpoint_field, selection_field, _ = TASK_KEY_FIELDS[task]
    selected_id = config.get(selection_field, "")
    provider = provider_for_url(config.get(endpoint_field, ""))
    for profile in config.get("api_key_profiles", []):
        if profile["id"] == selected_id and profile["provider"] == provider:
            return _clean_key(profile["key"])
    return ""


def _clean_key(key: str) -> str:
    """Reject header control characters without changing the stored credential."""
    key = key.strip()
    return key if not any(char in key for char in "\r\n\x00") else ""


def migrate_api_keys(config: dict[str, Any]) -> bool:
    """Preserve inline keys as profiles once, then remove the duplicate credential fields."""
    if not isinstance(config["api_key_profiles"], list):
        raise ValueError("API key profiles must be a list.")
    profiles = list(config["api_key_profiles"])
    ids: set[str] = set()
    for profile in profiles:
        if (not isinstance(profile, dict)
                or any(not isinstance(profile.get(field), str) for field in ("id", "name", "provider", "key"))
                or not profile["id"] or profile["id"] in ids or profile["provider"] not in PROVIDER_NAMES):
            raise ValueError("Invalid API key profile: unique ID, name, provider and key are required.")
        ids.add(profile["id"])
    changed = False
    for task, (endpoint_field, selection_field, legacy_field) in TASK_KEY_FIELDS.items():
        if legacy_field not in config:
            continue
        legacy_key = config[legacy_field]
        if not isinstance(legacy_key, str):
            raise ValueError("Legacy API keys must be strings.")
        key = legacy_key.strip()
        if key:
            provider = provider_for_url(config[endpoint_field])
            profile = next((p for p in profiles if p["key"] == key and p["provider"] == provider), None)
            if profile is None:
                profile = {"id": uuid.uuid4().hex, "name": f"{PROVIDER_NAMES[provider]} ({task})", "provider": provider, "key": key}
                profiles.append(profile)
            if not config[selection_field]:
                config[selection_field] = profile["id"]
        del config[legacy_field]
        changed = True
    if changed:
        config["api_key_profiles"] = profiles
    return changed


class GroqKeyRotation:
    """Rotate transcription credentials on the GUI thread, before workers receive snapshots."""

    def __init__(self) -> None:
        """Keep only the last pool/selection and the next slot; no locks or disk writes."""
        self._pool: tuple[str, ...] = ()
        self._selected = ""
        self._next = 0
        self.last_profile_id = ""

    def next_key(self, config: Mapping[str, Any]) -> str:
        """Start with the chosen key, then rotate unique nonempty Groq keys per request."""
        selected = selected_api_key(config, "transcription")
        self.last_profile_id = config.get("transcription_key_profile_id", "") if selected else ""
        if not selected or not config.get("groq_key_rotation", False) or provider_for_url(config.get("api_endpoint", "")) != "groq":
            self._pool = ()
            return selected
        keys = tuple(dict.fromkeys(
            key for profile in config["api_key_profiles"]
            if profile["provider"] == "groq" and (key := _clean_key(profile["key"]))
        ))
        if keys != self._pool or selected != self._selected:
            self._pool, self._selected, self._next = keys, selected, keys.index(selected)
        key = keys[self._next]
        self.last_profile_id = next(profile["id"] for profile in config["api_key_profiles"]
                                    if profile["provider"] == "groq" and _clean_key(profile["key"]) == key)
        self._next = (self._next + 1) % len(keys)
        return key
