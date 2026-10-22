"""Proxy resolution + connectivity diagnostics + transcription-endpoint test.

Single responsibility: decide outbound routing and classify connection outcomes.
Pure logic (no Qt, no widgets) — the UI passes in the model/url/key and renders the result.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Tuple
from urllib.parse import urlparse

import httpx

from app.core.api_keys import auth_headers, is_local_endpoint
from app.services.http_transport import is_tls_failure, request
from app.services.netutil import (
    PX_PROXY_URL,
    build_proxies,
    generate_test_wav_bytes,
    is_px_running,
    tcp_check,
)


def resolve_proxies(proxy_url: str, use_px: bool) -> Optional[Dict[str, str]]:
    """Explicit proxy URL wins; else a running local px proxy (if opted in); else system/env."""
    proxy_url = (proxy_url or "").strip()
    if proxy_url:
        return build_proxies(proxy_url)
    if use_px and is_px_running():
        logging.info("Routing requests through local px proxy at %s", PX_PROXY_URL)
        return build_proxies(PX_PROXY_URL)
    return None


def proxies_for(config: Mapping[str, Any]) -> Optional[Dict[str, str]]:
    """Proxies for a config snapshot (saved config, or the current settings form)."""
    return resolve_proxies(str(config.get("proxy_url", "")), bool(config.get("use_local_px_proxy", False)))


def diagnose_connectivity(api_url: str, proxies: Optional[Dict[str, str]]) -> str:
    """'blocked' (internet up but API unreachable) vs 'no_internet' (no outbound route).

    Behind a configured proxy the public hosts are not reachable directly, so the proxy
    itself is probed: a listening proxy means the route exists and the API is blocked.
    """
    proxy = urlparse((proxies or {}).get("https") or (proxies or {}).get("http") or "")
    if proxy.hostname:
        port = proxy.port or (443 if proxy.scheme == "https" else 80)
        return "blocked" if tcp_check(proxy.hostname, port) else "no_internet"
    raw_internet = tcp_check("1.1.1.1", 53) or tcp_check("8.8.8.8", 53)
    host = urlparse(api_url).hostname
    api_direct = tcp_check(host, 443)
    return "blocked" if (raw_internet or api_direct) else "no_internet"


#: Longest raw response body quoted in an API error message.
MAX_ERROR_DETAIL_CHARS = 1200


def api_error_details(response: httpx.Response) -> str:
    """Bounded diagnostics of an OpenAI-compatible error response, without the HTTP status.

    The ``error.message`` (with parameter, code and type) when the body is JSON, else the
    truncated body or the reason phrase; a request id follows when the server sent one.
    """
    lines = []
    try:
        payload = response.json()
    except ValueError:  # json.JSONDecodeError
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
        elif response.reason_phrase:
            lines.append(response.reason_phrase)

    request_id = response.headers.get("x-request-id")
    if request_id:
        lines.append(f"Request ID: {request_id}")
    return "\n".join(lines)


def format_api_error(response: httpx.Response) -> str:
    """``api_error_details`` headed by the HTTP status line."""
    details = api_error_details(response)
    return f"HTTP {response.status_code}" + (f"\n{details}" if details else "")


def classify_transport_error(error: Exception, api_url: str, proxies: Optional[Dict[str, str]]) -> str:
    """Map a failed request to 'proxy', 'ssl', 'blocked', 'no_internet', 'local_down' or 'unknown'."""
    if is_local_endpoint(api_url) and isinstance(error, (httpx.ConnectError, httpx.TimeoutException)):
        # This computer is reached directly, never through a proxy: nothing listens at that address.
        return "local_down"
    if isinstance(error, httpx.ProxyError):
        return "proxy"
    if is_tls_failure(error):
        return "ssl"
    if isinstance(error, (httpx.ConnectError, httpx.TimeoutException)):
        # An unreachable configured proxy fails at connect time, before the API is involved.
        return "proxy" if proxies and isinstance(error, httpx.ConnectError) else diagnose_connectivity(api_url, proxies)
    if isinstance(error, httpx.TransportError):
        return diagnose_connectivity(api_url, proxies)
    return "unknown"


def run_transcription_connection_test(api_url: str, api_key: str, model: str,
                                      proxies: Optional[Dict[str, str]]) -> Tuple[str, str]:
    """Send a tiny test audio to the endpoint and classify the outcome.

    Returns (result_key, detail) where result_key is one of:
    'ok', 'bad_key', 'bad_url', 'proxy', 'proxy_auth', 'ssl', 'reachable',
    'blocked', 'no_internet', 'unknown'.
    """
    headers = auth_headers(api_key)
    data = {"model": model}
    files = {"file": ("whispertyper_test.wav", generate_test_wav_bytes(), "audio/wav")}

    try:
        response = request(
            "POST", api_url, headers=headers, files=files, data=data, proxies=proxies, timeout=15
        )
    except Exception as e:
        return (classify_transport_error(e, api_url, proxies), str(e))

    status = response.status_code
    body = (response.text or "")[:400]
    if status == 200:
        return ("ok", body)
    if status in (401, 403):
        return ("bad_key", f"HTTP {status}: {body}")
    if status == 407:
        return ("proxy_auth", f"HTTP {status}: {body}")
    if status in (404, 405):
        # Server reached, but the endpoint path does not exist / does not accept POST —
        # almost always a misconfigured API endpoint URL (e.g. missing "/transcriptions").
        return ("bad_url", f"HTTP {status}: {body}")
    # Any other response: reached & answered (connection/proxy fine), request just rejected.
    return ("reachable", f"HTTP {status}: {body}")
