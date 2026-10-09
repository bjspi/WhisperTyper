"""Auth-free transport warming on a daemon; recording/upload never wait for it."""
from __future__ import annotations

import queue
import threading
import time
from urllib.parse import urlsplit, urlunsplit

import requests

from app.services.http_transport import request
from app.services.net import resolve_proxies


def warm_url(endpoint: str) -> str:
    """Use a safe HEAD target, stripping credentials, query and inference path."""
    parts = urlsplit(endpoint)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return ""
    host = parts.hostname
    authority = f"[{host}]" if ":" in host else host
    if parts.port:
        authority += f":{parts.port}"
    path = "/openai/v1/models" if host == "api.groq.com" else "/v1/models" if host == "api.openai.com" else "/"
    return urlunsplit((parts.scheme, authority, path, "", ""))


def _no_auth(prepared: requests.PreparedRequest) -> requests.PreparedRequest:
    """Explicit auth disables requests' implicit netrc origin credentials."""
    return prepared


class HttpWarmup:
    """Coalesce GUI snapshots and throttle each origin/route to once per 20 seconds."""

    def __init__(self) -> None:
        """Start one daemon independently of recording and inference workers."""
        self._queue: queue.SimpleQueue[tuple[tuple[str, ...], str, bool] | None] = queue.SimpleQueue()
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name="HttpWarmup", daemon=True)
        self._thread.start()

    def schedule(self, endpoints: tuple[str, ...], proxy_url: str, use_px: bool) -> None:
        """Enqueue only a URL/proxy snapshot; DNS/proxy detection happen off the GUI."""
        if not self._stopped:
            self._queue.put((endpoints, proxy_url, use_px))

    def close(self) -> None:
        """Stop without waiting for a network timeout on the GUI thread."""
        self._stopped = True
        self._queue.put(None)

    def _run(self) -> None:
        """Keep 401/404/405 connections too: any HTTP status proves transport reachability."""
        last: dict[tuple[str, str], float] = {}
        while not self._stopped:
            snapshot = self._queue.get()
            if snapshot is None:
                return
            while not self._queue.empty():
                snapshot = self._queue.get()
                if snapshot is None:
                    return
            endpoints, proxy_url, use_px = snapshot
            proxies = resolve_proxies(proxy_url, use_px)
            for endpoint in endpoints:
                if self._stopped:
                    return
                try:
                    url = warm_url(endpoint)
                    route = repr(proxies)  # Used only as a throttle key, never logged.
                    key = (url, route)
                    now = time.monotonic()
                    if not url or now - last.get(key, -float("inf")) < 20:
                        continue
                    last[key] = now
                    request("HEAD", url, stage="prewarm", auth=_no_auth, proxies=proxies,
                            timeout=(3, 3), allow_redirects=False)
                except (requests.RequestException, ValueError):
                    # Failure already has a transport trace; it must never block the actual request.
                    continue
