"""Chat-completion ("rephrasing") API client — one pure function, no Qt, no app state.

Used for every LLM round-trip in the app: LivePrompt instructions, generic post-transcription
rephrasing, the floating-window transformations, and the settings connection test. Callers
pass explicit values (snapshotted from the config on the GUI thread), so this stays safe to
run from any worker thread.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

from app.core.redaction import redact_for_log

#: Read timeout for chat completions; generous enough for long generations.
REQUEST_TIMEOUT_S = 30.0
MAX_ERROR_DETAIL_CHARS = 1200


class RephrasingError(RuntimeError):
    """The rephrasing API request failed (network error or non-2xx response)."""


def _format_api_error(response: requests.Response) -> str:
    """Return useful, bounded diagnostics from an OpenAI-compatible error response."""
    lines = [f"HTTP {response.status_code}"]
    try:
        payload = response.json()
    except (ValueError, requests.exceptions.JSONDecodeError):
        payload = None

    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        message = str(error.get("message") or "").strip()
        if message:
            lines.append(message)
        for label, key in (("Parameter", "param"), ("Code", "code"), ("Type", "type")):
            value = error.get(key)
            if value not in (None, ""):
                lines.append(f"{label}: {value}")
    elif error not in (None, ""):
        lines.append(str(error))
    else:
        body = (response.text or "").strip()
        if body:
            if len(body) > MAX_ERROR_DETAIL_CHARS:
                body = body[:MAX_ERROR_DETAIL_CHARS].rstrip() + "…"
            lines.append(body)
        elif response.reason:
            lines.append(str(response.reason))

    request_id = response.headers.get("x-request-id")
    if request_id:
        lines.append(f"Request ID: {request_id}")
    return "\n".join(lines)


def rephrase_text(
    *,
    system_prompt: str,
    user_prompt: str,
    api_url: str,
    api_key: str,
    model: str,
    temperature: float,
    context: str = "",
    proxies: Optional[Dict[str, str]] = None,
    timeout: float = REQUEST_TIMEOUT_S,
) -> str:
    """Send prompts to an OpenAI-compatible chat-completions endpoint and return the reply.

    Args:
        system_prompt: System-level instruction (omitted from the request when empty).
        user_prompt: The user's instruction or the text to transform.
        api_url: Chat-completions endpoint URL.
        api_key: Bearer token for the endpoint.
        model: Model identifier.
        temperature: Sampling temperature.
        context: Optional selected-text context, prepended to the user prompt.
        proxies: Optional ``requests`` proxies mapping.
        timeout: Request timeout in seconds.

    Returns:
        The model's reply text (stripped). May be empty if the model returned nothing.

    Raises:
        ValueError: If url/key/model are incomplete.
        RephrasingError: If the request fails or the endpoint answers with an error status.
    """
    if not api_url or not api_key or not model:
        raise ValueError("Rephrasing API settings are incomplete.")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    final_user_prompt = user_prompt
    if context:
        # Prepend context to the user prompt for better visibility by the model.
        final_user_prompt = (
            "Use the following context if relevant:\n---CONTEXT---\n"
            f"{context}\n---END CONTEXT---\n\nUser instruction: {user_prompt}"
        )
    messages.append({"role": "user", "content": final_user_prompt})

    data: Dict[str, Any] = {"model": model, "messages": messages}
    # GPT-5.6 Chat Completions accepts only its default temperature (1). Omitting the field
    # preserves that default; sending the configurable 0.0-1.0 value would make every Luna,
    # Terra, Sol, or family-alias request fail with HTTP 400.
    if not model.strip().lower().startswith("gpt-5.6"):
        data["temperature"] = temperature
    log_data = {**data, "messages": [
        {**m, "content": redact_for_log(m.get("content", ""))} for m in messages
    ]}
    logging.debug(f"Rephrasing request data: {log_data}")

    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=timeout, proxies=proxies)
    except requests.RequestException as e:
        raise RephrasingError(f"Rephrasing API request failed: {e}") from e

    if not response.ok:
        raise RephrasingError(f"Rephrasing API request failed:\n{_format_api_error(response)}")

    result = response.json()
    # OpenAI/Groq style: result['choices'][0]['message']['content']
    reply = (result.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
    logging.info(f"Rephrasing result: {redact_for_log(reply)}")
    return reply
