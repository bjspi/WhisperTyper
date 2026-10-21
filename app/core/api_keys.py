"""Provider-scoped API key profiles and migration of the former inline credentials."""
from __future__ import annotations

import ipaddress
import logging
import uuid
from typing import Any, Mapping
from urllib.parse import urlsplit

#: Per task: effective endpoint field, explicit key-profile field, legacy inline key field.
TASK_KEY_FIELDS = {
    "transcription": ("api_endpoint", "transcription_key_profile_id", "api_key"),
    "rephrasing": ("rephrasing_api_url", "rephrasing_key_profile_id", "rephrasing_api_key"),
}
PROVIDER_NAMES = {"openai": "OpenAI", "groq": "Groq", "custom": "Custom"}
#: Per task: the user's own endpoint, kept while an official provider is selected.
TASK_CUSTOM_URL_FIELDS = {"transcription": "transcription_custom_url", "rephrasing": "rephrasing_custom_url"}
_PROVIDER_API_BASES = {"openai": "https://api.openai.com/v1/", "groq": "https://api.groq.com/openai/v1/"}
_TASK_API_PATHS = {"transcription": "audio/transcriptions", "rephrasing": "chat/completions"}
# Official key prefixes; a mismatch is only a hint (keys are never rejected).
_KEY_PREFIXES = {"openai": "sk-", "groq": "gsk"}
#: Key selection of a custom endpoint that needs no key: requests go out without ``Authorization``.
NO_KEY_PROFILE_ID = "none"


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


def is_local_endpoint(url: str) -> bool:
    """True for an endpoint on this computer (localhost, 127.0.0.0/8, ::1)."""
    try:
        host = (urlsplit(url.strip()).hostname or "").rstrip(".").lower()
    except ValueError:
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def auth_headers(api_key: str) -> dict[str, str]:
    """The bearer header, or none for an endpoint that needs no key."""
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def provider_endpoint(provider: str, task: str) -> str:
    """Official endpoint of ``provider`` for ``task``; empty for a custom provider."""
    base = _PROVIDER_API_BASES.get(provider)
    return base + _TASK_API_PATHS[task] if base else ""


def key_format_warning(provider: str, key: str) -> str | None:
    """Translation key of a hint when ``key`` lacks the provider's usual prefix, else None."""
    prefix = _KEY_PREFIXES.get(provider)
    if prefix is None or key.strip().lower().startswith(prefix):
        return None
    return f"validation_{provider}_key_prefix"


def usable_profile_ids(profiles: list[dict[str, str]], provider: str) -> list[str]:
    """IDs of the profiles that hold a sendable key for ``provider``."""
    return [profile["id"] for profile in profiles if profile["provider"] == provider and _clean_key(profile["key"])]


def transcription_configured(config: Mapping[str, Any]) -> bool:
    """A transcription request can be sent: endpoint plus a matching selected key (or "no key needed")."""
    return bool(str(config.get("api_endpoint", "")).strip() and resolve_key_profile(config, "transcription"))


def rephrasing_configured(config: Mapping[str, Any]) -> bool:
    """A rephrasing request can be sent: endpoint, model and a matching selected key (or "no key needed")."""
    return bool(str(config.get("rephrasing_api_url", "")).strip()
                and str(config.get("rephrasing_model", "")).strip()
                and resolve_key_profile(config, "rephrasing"))


def resolve_key_profile(config: Mapping[str, Any], task: str) -> dict[str, str] | None:
    """The profile a task sends with: the explicit choice if it fits the endpoint's provider,
    otherwise the first profile of that provider with a sendable key (API Keys tab order).

    A custom endpoint only sends the key chosen for it: custom profiles are not tied to a host,
    so "the first custom key" may belong to a different server. Set to "no key needed", it
    resolves to a keyless profile with the ID :data:`NO_KEY_PROFILE_ID`.
    """
    endpoint_field, selection_field, _ = TASK_KEY_FIELDS[task]
    provider = provider_for_url(config.get(endpoint_field, ""))
    selected_id = config.get(selection_field, "")
    if provider == "custom" and selected_id == NO_KEY_PROFILE_ID:
        return {"id": NO_KEY_PROFILE_ID, "name": "", "provider": "custom", "key": ""}
    usable = [profile for profile in config.get("api_key_profiles", [])
              if profile["provider"] == provider and _clean_key(profile["key"])]
    chosen = next((profile for profile in usable if profile["id"] == selected_id), None)
    if chosen is not None or provider == "custom":
        return chosen
    return usable[0] if usable else None


def pin_custom_key_choices(config: dict[str, Any]) -> bool:
    """Make the key a custom endpoint used automatically its explicit choice (upgrade, once).

    Custom endpoints no longer fall back to the first custom key; a setup that relied on that
    keeps working with the same key, now visibly selected.
    """
    changed = False
    for endpoint_field, selection_field, _legacy in TASK_KEY_FIELDS.values():
        if provider_for_url(str(config.get(endpoint_field, ""))) != "custom" or config.get(selection_field, ""):
            continue
        first = next((profile for profile in config.get("api_key_profiles", [])
                      if profile["provider"] == "custom" and _clean_key(profile["key"])), None)
        if first is not None:
            config[selection_field] = first["id"]
            logging.info("Custom endpoint key pinned to profile %s (%s).", first["id"], selection_field)
            changed = True
    return changed


def selected_api_key(config: Mapping[str, Any], task: str) -> str:
    """The sendable key of the task's resolved profile, or "" when there is none (or none is needed)."""
    profile = resolve_key_profile(config, task)
    return _clean_key(profile["key"]) if profile else ""


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
        """Start with the resolved key, then rotate unique nonempty Groq keys per request."""
        profile = resolve_key_profile(config, "transcription")
        selected = _clean_key(profile["key"]) if profile else ""
        self.last_profile_id = profile["id"] if profile else ""
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
