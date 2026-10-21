"""Chat-completion ("rephrasing") API client — one pure function, no Qt, no app state.

Used for every LLM round-trip in the app: LivePrompt instructions, generic post-transcription
rephrasing, the floating-window transformations, and the settings connection test. Callers
pass explicit values (snapshotted from the config on the GUI thread), so this stays safe to
run from any worker thread.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from app.core.api_keys import auth_headers
from app.core.models import rephrasing_supports_temperature
from app.core.redaction import redact_for_log
from app.core.timing import NO_TIMING, OperationTiming
from app.services.http_transport import TimeoutValue, request
from app.services.net import format_api_error

#: Chat completions: connecting and sending are quick, but reasoning models can take minutes
#: to answer, so only the read phase gets a long budget.
REQUEST_TIMEOUT = httpx.Timeout(connect=15.0, read=120.0, write=30.0, pool=15.0)


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
    timeout: TimeoutValue = REQUEST_TIMEOUT,
    timing: OperationTiming = NO_TIMING,
) -> str:
    """Send prompts to an OpenAI-compatible chat-completions endpoint and return the reply.

    Args:
        system_prompt: System-level instruction (omitted from the request when empty).
        user_prompt: The user's instruction or the text to transform.
        api_url: Chat-completions endpoint URL.
        api_key: Bearer token for the endpoint ("" for an endpoint that needs no key).
        model: Model identifier.
        temperature: Sampling temperature.
        context: Optional selected-text context, prepended to the user prompt.
        proxies: Optional ``{"http": url, "https": url}`` proxies mapping.
        timeout: Request timeout (seconds, a ``(send, read)`` pair or an ``httpx.Timeout``).
        timing: Operation timings; the caller owns completion and delivery.

    Returns:
        The model's reply text (stripped). May be empty if the model returned nothing.

    Raises:
        ValueError: If url/model are incomplete.
        RephrasingError: If the request fails or the endpoint answers with an error status.
    """
    if not api_url or not model:
        raise ValueError("Rephrasing API settings are incomplete.")

    headers = {**auth_headers(api_key), "Content-Type": "application/json"}

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
    if rephrasing_supports_temperature(model):
        data["temperature"] = temperature
    log_data = {**data, "messages": [
        {**m, "content": redact_for_log(m.get("content", ""))} for m in messages
    ]}
    logging.debug(f"Rephrasing request data: {log_data}")

    try:
        timing.mark("rephrase_request_start")
        response = request("POST", api_url, timing=timing, stage="rephrase", headers=headers, json=data,
                           timeout=timeout, proxies=proxies)
        timing.mark("rephrase_response_received")
    except httpx.HTTPError as e:
        raise RephrasingError(f"Rephrasing API request failed: {e}") from e

    if not response.is_success:
        raise RephrasingError(f"Rephrasing API request failed:\n{format_api_error(response)}")

    try:
        result = response.json()
    except ValueError as e:  # json.JSONDecodeError
        raise RephrasingError("Rephrasing API answered without a JSON body.") from e
    # OpenAI/Groq style: result['choices'][0]['message']['content']
    choices = result.get("choices") if isinstance(result, dict) else None
    if not choices:
        raise RephrasingError("Rephrasing API answered with an empty response (no choices).")
    message = (choices[0] or {}).get("message") or {}
    reply = str(message.get("content") or "").strip()
    timing.mark("rephrase_response_parsed")
    logging.info(f"Rephrasing result: {redact_for_log(reply)}")
    return reply
