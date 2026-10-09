"""Provider-scoped API key profiles and migration of the former inline credentials."""
from __future__ import annotations

import logging
import uuid
from typing import Any, Mapping
from urllib.parse import urlsplit

TASK_KEY_FIELDS = {
    "transcription": ("api_endpoint", "transcription_key_profile_id", "api_key"),
    "rephrasing": ("rephrasing_api_url", "rephrasing_key_profile_id", "rephrasing_api_key"),
}
PROVIDER_NAMES = {"openai": "OpenAI", "groq": "Groq", "custom": "Custom"}
_PROVIDER_API_BASES = {"openai": "https://api.openai.com/v1/", "groq": "https://api.groq.com/openai/v1/"}
_TASK_API_PATHS = {"transcription": "audio/transcriptions", "rephrasing": "chat/completions"}


def masked_api_key(key: str) -> str:
    """Show recognizable ends without revealing an entire short credential."""
    key = key.strip()
    return key[:10] + "••••••" + key[-4:] if len(key) > 14 else "•" * len(key)


def provider_for_url(url: str) -> str:
    """Recognize official provider hosts, without accepting lookalike domains."""
    try:
        host = urlsplit(url.strip()).hostname
    except ValueError:
        return "custom"
    return {"api.openai.com": "openai", "api.groq.com": "groq"}.get(host or "", "custom")


def provider_endpoint(provider: str, task: str) -> str:
    """Official endpoint of ``provider`` for ``task``; empty for a custom provider."""
    base = _PROVIDER_API_BASES.get(provider)
    return base + _TASK_API_PATHS[task] if base else ""


def usable_profile_ids(profiles: list[dict[str, str]], provider: str) -> list[str]:
    """IDs of the profiles that hold a sendable key for ``provider``."""
    return [profile["id"] for profile in profiles if profile["provider"] == provider and _clean_key(profile["key"])]


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


def _sanitize_profiles(raw: Any) -> tuple[list[dict[str, str]], bool]:
    """Repair hand-edited profiles instead of failing startup; logs indices, never contents."""
    if not isinstance(raw, list):
        logging.warning("api_key_profiles is not a list; starting with no key profiles.")
        return [], True
    profiles: list[dict[str, str]] = []
    ids: set[str] = set()
    changed = False
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or not isinstance(item.get("key"), str):
            logging.warning("Dropping invalid API key profile #%s (not an object or no key).", index)
            changed = True
            continue
        fields: dict[str, str] = {}
        for field in ("id", "name", "provider"):
            value = item.get(field)
            fields[field] = value if isinstance(value, str) else ""
        provider = fields["provider"].strip().lower()
        profile = {
            "id": fields["id"] if fields["id"] and fields["id"] not in ids else uuid.uuid4().hex,
            "provider": provider if provider in PROVIDER_NAMES else "custom",
            "key": item["key"],
        }
        profile["name"] = fields["name"].strip() or PROVIDER_NAMES[profile["provider"]]
        profile = {field: profile[field] for field in ("id", "name", "provider", "key")}
        if profile != item:
            logging.warning("Repaired API key profile #%s (id, name or provider).", index)
            changed = True
        ids.add(profile["id"])
        profiles.append(profile)
    return profiles, changed


def migrate_api_keys(config: dict[str, Any]) -> bool:
    """Repair profiles, preserve inline keys as profiles once, then drop the duplicate credential fields."""
    profiles, changed = _sanitize_profiles(config.get("api_key_profiles"))
    for task, (endpoint_field, selection_field, legacy_field) in TASK_KEY_FIELDS.items():
        if legacy_field not in config:
            continue
        legacy_key = config[legacy_field]
        key = legacy_key.strip() if isinstance(legacy_key, str) else ""
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
