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


class RephrasingError(RuntimeError):
    """The rephrasing API request failed (network error or non-2xx response)."""


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

    data: Dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
    log_data = {**data, "messages": [
        {**m, "content": redact_for_log(m.get("content", ""))} for m in messages
    ]}
    logging.debug(f"Rephrasing request data: {log_data}")

    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=timeout, proxies=proxies)
        response.raise_for_status()
    except Exception as e:
        raise RephrasingError(f"Rephrasing API request failed: {e}") from e

    result = response.json()
    # OpenAI/Groq style: result['choices'][0]['message']['content']
    reply = (result.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
    logging.info(f"Rephrasing result: {redact_for_log(reply)}")
    return reply
