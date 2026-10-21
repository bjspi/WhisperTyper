"""Which settings a save changed, for one log line without credentials or user texts. No Qt here."""
from __future__ import annotations

from typing import Any, Mapping
from urllib.parse import urlsplit

#: Settings whose values are never logged: credentials, proxy URLs (may hold a password) and
#: the user's own texts (prompts, replacements, prompt entries).
_PRIVATE_KEYS = frozenset({
    "api_key_profiles", "api_key", "rephrasing_api_key", "proxy_url", "prompt",
    "transcription_second_pass_prompt", "replacements_rules", "post_rephrasing_entries",
})
#: Endpoint URLs are logged by host only (a URL may carry credentials in its user info or query).
_URL_KEYS = frozenset({"api_endpoint", "rephrasing_api_url", "transcription_custom_url", "rephrasing_custom_url"})
#: Longer strings are reported as changed, without the value.
_MAX_VALUE_CHARS = 80


def describe_changes(before: Mapping[str, Any], after: Mapping[str, Any]) -> str:
    """``key=old->new`` for every changed setting; private or long values only as ``key=changed``."""
    parts = []
    for key in sorted(set(before) | set(after)):
        old, new = before.get(key), after.get(key)
        if old == new:
            continue
        if key in _URL_KEYS:
            old, new = _host(old), _host(new)
        if key in _PRIVATE_KEYS or not (_loggable(old) and _loggable(new)):
            parts.append(f"{key}=changed")
        else:
            parts.append(f"{key}={old!r}->{new!r}")
    return " ".join(parts) or "none"


def _loggable(value: Any) -> bool:
    if isinstance(value, str):
        return len(value) <= _MAX_VALUE_CHARS
    return value is None or isinstance(value, (bool, int, float))


def _host(url: Any) -> str:
    try:
        return f"host:{urlsplit(str(url or '')).hostname or ''}"
    except ValueError:
        return "host:?"
