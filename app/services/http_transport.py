"""Shared httpx clients timed through httpcore's official ``trace`` extension.

No transport internals are subclassed: connection, TLS, proxy-tunnel and HTTP/1.1 phases come
from the documented trace callback, exchange boundaries (redirects) from client event hooks.
Transport logs carry routing metadata only — never credentials, queries or payloads.
"""
from __future__ import annotations

import http.cookiejar
import itertools
import os
import socket
import ssl
import threading
import time
import urllib.request
import weakref
from typing import Any, Optional, Union
from urllib.parse import urlsplit

import httpx

from app.core.timing import NO_TIMING, OperationTiming, queue_http_timing

#: Idle pooled connections are discarded after this long. Routers and NATs silently drop idle
#: connections after a few minutes; reusing such a connection hangs until the read timeout, so a
#: fresh one is opened instead (the warmup re-touches active origins every 20 s).
KEEPALIVE_EXPIRY_S = 60.0
_LIMITS = httpx.Limits(max_connections=None, max_keepalive_connections=8, keepalive_expiry=KEEPALIVE_EXPIRY_S)
# OS-level keepalive on top of httpcore's built-in TCP_NODELAY.
_SOCKET_OPTIONS = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]

_CLIENTS: dict[tuple[Optional[str], Any], httpx.Client] = {}
_CLIENTS_LOCK = threading.Lock()
_OPERATION_IDS = itertools.count(1)
_CONNECTION_IDS = itertools.count(1)
# Final (post-TLS) network stream -> stable connection number for reuse diagnostics.
_STREAM_IDS: "weakref.WeakKeyDictionary[Any, int]" = weakref.WeakKeyDictionary()

TimeoutValue = Union[None, float, tuple[float, float], httpx.Timeout]


def is_tls_failure(error: BaseException) -> bool:
    """True if a transport error was caused by TLS (certificate/handshake) failure."""
    current: Optional[BaseException] = error
    while current is not None:
        if isinstance(current, ssl.SSLError):
            return True
        current = current.__cause__ or current.__context__
    return False


def _error_name(error: BaseException) -> str:
    """Class name for logs; TLS failures are named explicitly instead of a generic ConnectError."""
    return "SSLError" if is_tls_failure(error) else type(error).__name__


class _Exchange:
    """One wire request/response; a redirect starts the next exchange of the same call."""

    def __init__(self, call: "_CallTrace", request: httpx.Request, start_ns: int) -> None:
        """Capture public routing metadata only."""
        self.call = call
        self.number = len(call.exchanges) + 1
        self.events: dict[str, int] = {"start": start_ns}
        self.response: Optional[httpx.Response] = None
        self.metadata: dict[str, Any] = {
            "op": call.operation_id, "stage": call.stage, "exchange": self.number,
            "method": request.method, "host": request.url.host, "route": "proxy" if call.proxied else "direct",
            "connection": "unknown", "reused": "unknown", "status": "none",
            "upload_bytes": int(request.headers.get("content-length", 0) or 0), "response_body_bytes": 0,
        }
        self.finished = False

    def mark(self, phase: str, at_ns: Optional[int] = None) -> None:
        """Record once and mirror the milestone onto the owning operation."""
        if phase in self.events or self.finished:
            return
        now = at_ns if at_ns is not None else time.perf_counter_ns()
        self.events[phase] = now
        suffix = "" if self.number == 1 else f"_{self.number}"
        self.call.timing.mark(f"{self.call.stage}_http_{phase}{suffix}", now)

    def finish(self, error: str = "none") -> None:
        """Queue the exchange with partial measurements on failure; never exception text."""
        if self.finished:
            return
        self.metadata["last_phase"] = next(reversed(self.events))
        if self.response is not None and self.response.is_closed:
            self.metadata["response_body_bytes"] = self.response.num_bytes_downloaded
        self.mark("end")
        self.metadata["error"] = error
        self.finished = True
        queue_http_timing(self.metadata, self.events)


class _CallTrace:
    """httpx ``trace`` callback for one logical API call (all its redirects)."""

    def __init__(self, timing: OperationTiming, stage: str, proxied: bool) -> None:
        """Bind milestones to the caller's operation, or to a transport-only id."""
        self.timing = timing
        self.stage = stage
        self.proxied = proxied
        self.operation_id = timing.operation_id if timing is not NO_TIMING else f"http{next(_OPERATION_IDS)}"
        self.start_ns = time.perf_counter_ns()
        self.exchanges: list[_Exchange] = []
        # Only ``started`` events carry the request; remember per step whether it is the proxy CONNECT.
        self._tunnel_steps: dict[str, bool] = {}

    @property
    def current(self) -> Optional[_Exchange]:
        """The exchange currently on the wire."""
        return self.exchanges[-1] if self.exchanges else None

    def begin(self, request: httpx.Request) -> None:
        """Client request hook: close the previous hop and open a fresh exchange."""
        if self.current is not None:
            self.current.finish()
        start = self.start_ns if not self.exchanges else time.perf_counter_ns()
        self.exchanges.append(_Exchange(self, request, start))
        self.exchanges[-1].mark("adapter_start")

    def response(self, response: httpx.Response) -> None:
        """Client response hook: status and pooled-connection identity."""
        exchange = self.current
        if exchange is None:
            return
        exchange.response = response
        exchange.metadata["status"] = response.status_code
        stream = response.extensions.get("network_stream")
        if stream is not None:
            try:
                if stream not in _STREAM_IDS:
                    _STREAM_IDS[stream] = next(_CONNECTION_IDS)
                exchange.metadata["connection"] = _STREAM_IDS[stream]
            except TypeError:
                pass  # Stream type without weakref support: identity stays "unknown".

    def __call__(self, name: str, info: dict[str, Any]) -> None:
        """Translate httpcore trace events (``<area>.<step>.<started|complete|failed>``)."""
        exchange = self.current
        if exchange is None:
            return
        step, _, state = name.rpartition(".")
        if state == "started":
            request = info.get("request")
            self._tunnel_steps[step] = request is not None and request.method == b"CONNECT"
        tunnel = self._tunnel_steps.get(step, False)
        if step == "connection.connect_tcp":
            if state == "started":
                exchange.metadata["reused"] = False
                exchange.mark("connect_start")
                exchange.mark("tcp_start")  # Includes DNS: httpcore resolves inside connect_tcp.
            elif state == "complete":
                exchange.mark("tcp_end")
        elif step in ("connection.start_tls", "proxy.start_tls"):
            # The first TLS on a proxied route is to the proxy itself; proxy.start_tls is the origin.
            prefix = "proxy_tls" if step == "connection.start_tls" and self.proxied else "tls"
            if state == "started":
                exchange.mark(f"{prefix}_start")
            elif state == "complete":
                exchange.mark(f"{prefix}_end")
        elif step == "http11.send_request_headers":
            if tunnel:
                if state == "started":
                    exchange.mark("proxy_tunnel_start")
            elif state == "started":
                if exchange.metadata["reused"] == "unknown":
                    exchange.metadata["reused"] = True
                if "connect_start" in exchange.events:
                    exchange.mark("connect_end")
            elif state == "complete":
                exchange.mark("headers_sent")
        elif step == "http11.send_request_body" and not tunnel:
            if state == "started":
                exchange.mark("upload_start")
            elif state == "complete":
                exchange.mark("request_sent")
        elif step == "http11.receive_response_headers" and state == "complete":
            if tunnel:
                exchange.mark("proxy_tunnel_end")
            else:
                # httpcore reports the parsed status line + headers; this is the first-byte milestone.
                now = time.perf_counter_ns()
                exchange.mark("first_byte", now)
                exchange.mark("headers_received", now)
        elif step == "http11.receive_response_body" and state == "complete" and not tunnel:
            exchange.mark("body_end")

    def finish(self, error: str = "none") -> None:
        """Close every exchange of this call exactly once."""
        for exchange in self.exchanges:
            exchange.finish(error if exchange is self.current else "none")


def _on_request(request: httpx.Request) -> None:
    """Event hook shared by all clients: each (redirected) request is a new exchange."""
    tracer = request.extensions.get("trace")
    if isinstance(tracer, _CallTrace):
        tracer.begin(request)


def _on_response(response: httpx.Response) -> None:
    """Event hook shared by all clients."""
    tracer = response.request.extensions.get("trace")
    if isinstance(tracer, _CallTrace):
        tracer.response(response)


def _ssl_context(verify: Union[bool, str]) -> Union[bool, ssl.SSLContext]:
    """Verify with certifi + SSL_CERT_FILE/DIR like httpx; REQUESTS_CA_BUNDLE stays honored."""
    if verify is False:
        return False
    if isinstance(verify, str):
        return ssl.create_default_context(cafile=verify)
    bundle = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("CURL_CA_BUNDLE")
    if bundle:
        return ssl.create_default_context(cafile=bundle)
    return httpx.create_ssl_context(trust_env=True)


def _no_cookie_jar() -> http.cookiejar.CookieJar:
    """API calls are stateless: a jar whose policy accepts no domain never stores Set-Cookie."""
    return http.cookiejar.CookieJar(http.cookiejar.DefaultCookiePolicy(allowed_domains=[]))


def _client(proxy: Optional[str], verify: Union[bool, str]) -> httpx.Client:
    """One pooled, thread-safe client per route; no retries, no cookie persistence, no netrc."""
    key = (proxy, verify)
    with _CLIENTS_LOCK:
        client = _CLIENTS.get(key)
        if client is None:
            context = _ssl_context(verify)
            # An HTTPS proxy is verified like the origin; httpx rejects a context for plain-HTTP proxies.
            proxy_context = context if isinstance(context, ssl.SSLContext) and proxy and proxy.startswith("https:") else None
            transport = httpx.HTTPTransport(
                verify=context, limits=_LIMITS, retries=0, socket_options=_SOCKET_OPTIONS,
                proxy=httpx.Proxy(proxy, ssl_context=proxy_context) if proxy else None,
            )
            client = httpx.Client(
                transport=transport, trust_env=False, timeout=None, cookies=_no_cookie_jar(),
                event_hooks={"request": [_on_request], "response": [_on_response]},
            )
            _CLIENTS[key] = client
        return client


def _select_proxy(url: str, proxies: Optional[dict[str, str]]) -> Optional[str]:
    """Explicit mapping first; otherwise the system/env proxy unless the host is bypassed."""
    parts = urlsplit(url)
    if proxies:
        return proxies.get(parts.scheme) or proxies.get("all")
    host = parts.hostname or ""
    system = urllib.request.getproxies()
    if not system or urllib.request.proxy_bypass(host):
        return None
    return system.get(parts.scheme) or system.get("all")


def _timeout(value: TimeoutValue) -> httpx.Timeout:
    """A ``(send, read)`` pair bounds connect + upload by the first value, the response by the second."""
    if isinstance(value, httpx.Timeout):
        return value
    if isinstance(value, tuple):
        send, read = value
        return httpx.Timeout(connect=send, write=send, read=read, pool=send)
    return httpx.Timeout(value)


def request(method: str, url: str, *, timing: OperationTiming = NO_TIMING, stage: str = "connection_test",
            proxies: Optional[dict[str, str]] = None, timeout: TimeoutValue = None,
            verify: Union[bool, str] = True, follow_redirects: bool = True, **kwargs: Any) -> httpx.Response:
    """Send one logical API call over the shared pools; the body is always read before returning.

    ``proxies`` uses the ``{"http": url, "https": url}`` mapping from ``services.net``.
    Remaining keyword arguments (headers, data, files, json, auth) go to ``httpx.Client.request``.
    """
    proxy = _select_proxy(url, proxies)
    tracer = _CallTrace(timing, stage, proxied=proxy is not None)
    try:
        return _client(proxy, verify).request(
            method, url, timeout=_timeout(timeout), follow_redirects=follow_redirects,
            extensions={"trace": tracer}, **kwargs,
        )
    except Exception as error:
        tracer.finish(_error_name(error))
        raise
    finally:
        tracer.finish()


def close_transport() -> None:
    """Release pooled connections at shutdown (and between tests); clients are rebuilt lazily."""
    with _CLIENTS_LOCK:
        clients = list(_CLIENTS.values())
        _CLIENTS.clear()
    for client in clients:
        client.close()
