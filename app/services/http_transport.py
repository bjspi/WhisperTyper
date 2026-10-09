"""Shared HTTP/1.1 pools and socket milestones, without credentials in transport logs."""
from __future__ import annotations

import http.client
import itertools
import socket
import sys
import threading
import time
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlsplit

import requests
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from urllib3.exceptions import ConnectTimeoutError, LocationParseError, NameResolutionError, NewConnectionError
from urllib3.util.connection import allowed_gai_family
from urllib3.util.timeout import _DEFAULT_TIMEOUT

from app.core.timing import OperationTiming, queue_http_timing

_LOCAL = threading.local()
_CONNECTION_IDS = itertools.count(1)


class _Trace:
    """One wire request, including each redirect, owned by its calling thread."""

    def __init__(self, method: str, url: str, context: dict[str, Any], proxy: bool) -> None:
        """Capture public routing metadata only, with a fresh span for each exchange."""
        self.timing = context["timing"]
        self.stage = context["stage"]
        self.number = len(context["traces"]) + 1
        self.events = {"start": context["start"] if self.number == 1 else time.perf_counter_ns()}
        self.metadata = {
            "op": self.timing.operation_id if self.timing else context["id"],
            "stage": self.stage, "exchange": self.number, "method": method,
            "host": urlsplit(url).hostname, "route": "proxy" if proxy else "direct",
            "connection": "unknown", "reused": "unknown", "status": "none",
            "upload_bytes": 0, "response_body_bytes": 0,
        }
        self.finished = False
        context["traces"].append(self)
        self.mark("adapter_start")

    def mark(self, phase: str) -> None:
        """Capture once; enqueue operation milestones without log-handler calls."""
        if phase in self.events or self.finished:
            return
        now = time.perf_counter_ns()
        self.events[phase] = now
        if self.timing:
            suffix = "" if self.number == 1 else f"_{self.number}"
            self.timing.mark(f"{self.stage}_http_{phase}{suffix}", now)

    def finish(self, error: str = "none") -> None:
        """Keep partial measurements on failure, without exception text or payloads."""
        if self.finished:
            return
        self.metadata["last_phase"] = next(reversed(self.events))
        self.mark("end")
        self.metadata["error"] = error
        self.finished = True
        queue_http_timing(self.metadata, self.events)


def _trace() -> Any:
    """Return only this thread's current exchange; connection pools remain shared."""
    return getattr(_LOCAL, "trace", None)


class _Response(http.client.HTTPResponse):
    """Observe the first response byte before parsing status/headers."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Keep the exchange trace after the adapter returns its streaming response."""
        super().__init__(*args, **kwargs)
        self.trace = _trace()

    def _read_status(self) -> Any:
        if self.trace and self.fp.peek(1):
            self.trace.mark("first_byte")
        return super()._read_status()  # type: ignore[misc]

    def begin(self) -> None:
        """Record headers separately from first byte and complete body."""
        super().begin()
        if self.trace:
            self.trace.metadata["status"] = self.status
            self.trace.mark("headers_received")


class _Connection(HTTPConnection):
    """Instrument the existing urllib3 connection without changing TLS verification."""

    response_class = _Response

    def _new_conn(self) -> socket.socket:
        """Split urllib3's DNS/TCP setup while preserving address fallback and errors."""
        trace = _trace()
        if trace:
            trace.metadata.update(connection=next(_CONNECTION_IDS), reused=False)
            self._transport_id = trace.metadata["connection"]
            trace.mark("dns_start")
        host = self._dns_host.strip("[]")
        try:
            try:
                host.encode("idna")
            except UnicodeError:
                raise LocationParseError("Invalid DNS name") from None
            addresses = socket.getaddrinfo(host, self.port, allowed_gai_family(), socket.SOCK_STREAM)
            if trace:
                trace.mark("dns_end")
                trace.mark("tcp_start")
            error: OSError = OSError("getaddrinfo returns an empty list")
            for family, kind, protocol, _, address in addresses:
                sock = None
                try:
                    sock = socket.socket(family, kind, protocol)
                    for option in self.socket_options or ():
                        sock.setsockopt(*option)
                    if self.timeout is not _DEFAULT_TIMEOUT:
                        sock.settimeout(self.timeout)
                    if self.source_address:
                        sock.bind(self.source_address)
                    sock.connect(address)
                    if trace:
                        trace.mark("tcp_end")
                        if isinstance(self, HTTPSConnection) and not self.proxy_is_tunneling:
                            trace.mark("proxy_tls_start" if self.proxy_is_forwarding else "tls_start")
                    sys.audit("http.client.connect", self, self.host, self.port)
                    return sock
                except OSError as exc:
                    error = exc
                    if sock is not None:
                        sock.close()
            raise error
        except socket.gaierror as exc:
            raise NameResolutionError(self.host, self, exc) from exc
        except socket.timeout as exc:
            raise ConnectTimeoutError(self, "TCP connection timed out") from exc
        except OSError as exc:
            raise NewConnectionError(self, "TCP connection failed") from exc

    def connect(self) -> None:
        """TLS setup includes handshake, CA loading and certificate verification."""
        trace = _trace()
        if trace:
            trace.mark("connect_start")
        super().connect()
        if trace:
            if isinstance(self, HTTPSConnection):
                trace.mark("proxy_tls_end" if self.proxy_is_forwarding else "tls_end")
            trace.mark("connect_end")

    def _tunnel(self) -> None:
        """Exclude the proxy's CONNECT response from the API's TTFB."""
        trace = _trace()
        if trace:
            trace.mark("proxy_tunnel_start")
        _LOCAL.trace = None
        try:
            super()._tunnel()
        finally:
            _LOCAL.trace = trace
        if trace:
            trace.mark("proxy_tunnel_end")
            trace.mark("tls_start")

    def _connect_tls_proxy(self, hostname: str, sock: socket.socket) -> Any:
        """Measure secure proxy setup independently of origin TLS."""
        trace = _trace()
        if trace:
            trace.mark("proxy_tls_start")
        result = super()._connect_tls_proxy(hostname, sock)  # type: ignore[misc]
        if trace:
            trace.mark("proxy_tls_end")
        return result

    def request(self, *args: Any, **kwargs: Any) -> None:
        """Bind socket identity to this exchange and mark completion of the send."""
        trace = _trace()
        if trace and self.sock is not None:
            trace.metadata.update(connection=getattr(self, "_transport_id", "unknown"), reused=True)
        super().request(*args, **kwargs)
        if trace:
            trace.mark("request_sent")

    def endheaders(self, *args: Any, **kwargs: Any) -> None:
        """Prevent request headers from being counted as uploaded audio."""
        self._sending_headers = True
        try:
            super().endheaders(*args, **kwargs)
        finally:
            self._sending_headers = False
        if _trace():
            _trace().mark("headers_sent")

    def send(self, data: Any) -> None:
        """Measure body sends only after any lazy socket connection finishes."""
        if self.sock is None and self.auto_open:
            self.connect()
        trace = _trace()
        body = not getattr(self, "_sending_headers", False)
        if trace and body:
            trace.mark("upload_start")
        super().send(data)
        if trace and body:
            trace.metadata["upload_bytes"] += len(data)


class _HTTPSConnection(_Connection, HTTPSConnection):
    """HTTPS with unchanged urllib3 certificate and proxy handling."""


class _HTTPPool(HTTPConnectionPool):
    """Shared HTTP pool."""

    ConnectionCls = _Connection


class _HTTPSPool(HTTPSConnectionPool):
    """Shared HTTPS pool."""

    ConnectionCls = _HTTPSConnection


class _Adapter(requests.adapters.HTTPAdapter):
    """Own pools across worker lifetimes; do not retry paid POSTs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Serialize only proxy-manager creation, never network IO or logging."""
        self._proxy_lock = threading.Lock()
        super().__init__(*args, **kwargs)

    def init_poolmanager(self, *args: Any, **kwargs: Any) -> None:
        """Enable OS keepalive without changing socket timeouts or TLS policy."""
        kwargs["socket_options"] = [*HTTPConnection.default_socket_options, (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
        super().init_poolmanager(*args, **kwargs)
        self.poolmanager.pool_classes_by_scheme = {"http": _HTTPPool, "https": _HTTPSPool}

    def proxy_manager_for(self, proxy: str, **kwargs: Any) -> Any:
        """Apply the same socket tracing and keepalive through configured proxies."""
        kwargs["socket_options"] = [*HTTPConnection.default_socket_options, (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
        with self._proxy_lock:
            manager = super().proxy_manager_for(proxy, **kwargs)
            # SOCKS requires its own connection implementation; requests' optional support remains intact.
            if not proxy.lower().startswith("socks"):
                manager.pool_classes_by_scheme = {"http": _HTTPPool, "https": _HTTPSPool}
            return manager

    def send(self, request: Any, *args: Any, **kwargs: Any) -> requests.Response:
        """Trace each wire exchange, including redirects, separately."""
        context = _LOCAL.context
        proxy = requests.utils.select_proxy(request.url, kwargs.get("proxies") or {})
        trace = _Trace(request.method, request.url, context, bool(proxy))
        _LOCAL.trace = trace
        try:
            response = super().send(request, *args, **kwargs)
            trace.metadata["status"] = response.status_code
            trace.mark("headers_received")
            original_stream = response.raw.stream

            def measured_stream(*stream_args: Any, **stream_kwargs: Any) -> Iterator[bytes]:
                """Measure consumed bytes for fixed-length, chunked and compressed bodies."""
                for chunk in original_stream(*stream_args, **stream_kwargs):
                    trace.metadata["response_body_bytes"] += len(chunk)
                    yield chunk
                trace.mark("body_end")
                trace.finish()

            response.raw.stream = measured_stream
            if request.method == "HEAD" or response.raw.length_remaining == 0:
                trace.mark("body_end")
                trace.finish()
            return response
        except Exception as exc:
            trace.finish(type(exc).__name__)
            raise
        finally:
            _LOCAL.trace = None


_ADAPTER = _Adapter(pool_connections=16, pool_maxsize=8, max_retries=0, pool_block=False)


def request(method: str, url: str, *, timing: OperationTiming | None = None,
            stage: str = "connection_test", **kwargs: Any) -> requests.Response:
    """Use shared pools with isolated headers/cookies for every logical API call."""
    context: dict[str, Any] = {"timing": timing, "stage": stage, "traces": [],
                               "start": time.perf_counter_ns(), "id": f"http{next(_CONNECTION_IDS)}"}
    _LOCAL.context = context
    session = requests.Session()
    session.mount("http://", _ADAPTER)
    session.mount("https://", _ADAPTER)
    kwargs["stream"] = False
    try:
        # Always consume the body so its connection returns to the shared pool.
        return session.request(method, url, **kwargs)
    except Exception as exc:
        for trace in context["traces"]:
            trace.finish(type(exc).__name__)
        raise
    finally:
        for trace in context["traces"]:
            trace.finish()
        _LOCAL.context = None
        # Session.close() would close the shared adapter; no Session state is reused.


def close_transport() -> None:
    """Release idle connections at application shutdown, after inference workers stop."""
    _ADAPTER.close()
