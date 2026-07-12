"""Proxy resolution + connectivity diagnostics + transcription-endpoint test.

Single responsibility: decide outbound routing and classify connection outcomes.
Pure logic (no Qt, no widgets) — the UI passes in the model/url/key and renders the result.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import requests

from app.core.netutil import (
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


def diagnose_connectivity(api_url: str, proxies: Optional[Dict[str, str]]) -> str:
    """'blocked' (internet up but API unreachable) vs 'no_internet' (no outbound route)."""
    raw_internet = tcp_check("1.1.1.1", 53) or tcp_check("8.8.8.8", 53)
    host = urlparse(api_url).hostname
    api_direct = tcp_check(host, 443)
    return "blocked" if (raw_internet or api_direct) else "no_internet"


def run_transcription_connection_test(api_url: str, api_key: str, model: str,
                                      proxies: Optional[Dict[str, str]]) -> Tuple[str, str]:
    """Send a tiny test audio to the endpoint and classify the outcome.

    Returns (result_key, detail) where result_key is one of:
    'ok', 'bad_key', 'bad_url', 'proxy', 'proxy_auth', 'ssl', 'reachable',
    'blocked', 'no_internet', 'unknown'.
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"model": model}
    files = {"file": ("whispertyper_test.wav", generate_test_wav_bytes(), "audio/wav")}

    try:
        response = requests.post(
            api_url, headers=headers, files=files, data=data, proxies=proxies, timeout=15
        )
    except requests.exceptions.ProxyError as e:
        return ("proxy", str(e))
    except requests.exceptions.SSLError as e:
        return ("ssl", str(e))
    except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError,
            requests.exceptions.Timeout) as e:
        return (diagnose_connectivity(api_url, proxies), str(e))
    except Exception as e:
        return ("unknown", str(e))

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
