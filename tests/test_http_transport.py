"""Real local sockets: pool reuse, phase timing, TLS/proxies, failures and warming."""
from __future__ import annotations

import gzip
import http.client
import logging
import select
import socket
import ssl
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import requests
from urllib3.exceptions import InsecureRequestWarning

from app.core.timing import OperationTiming, flush_timing_logs
from app.services import http_transport as transport
from app.services.http_warmup import HttpWarmup, _no_auth, warm_url

CERT = Path(__file__).parent / "fixtures" / "localhost-cert.pem"
KEY = Path(__file__).parent / "fixtures" / "localhost-key.pem"  # Public test fixture, never used outside localhost.


class API(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    seen = []
    cookies = []

    def do_HEAD(self):
        self.seen.append((self.command, self.client_address, self.headers.get("Authorization"), b""))
        self.send_response(401)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        self.cookies.append(self.headers.get("Cookie"))
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.seen.append((self.command, self.client_address, self.headers.get("Authorization"), body))
        if self.path == "/drop":
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            self.close_connection = True
            return
        if self.path == "/slow":
            time.sleep(0.08)
            self.wfile.write(b"HTTP/1.1 200 OK\r\n")
            self.wfile.flush()
            time.sleep(0.08)
            self.wfile.write(b"Content-Length: 2\r\n\r\n")
            self.wfile.flush()
            time.sleep(0.08)
            self.wfile.write(b"OK")
            self.wfile.flush()
            return
        if self.path == "/redirect":
            self.send_response(307)
            self.send_header("Location", "/ok")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path == "/chunked":
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self.wfile.write(b"2\r\nOK\r\n0\r\n\r\n")
            self.wfile.flush()
            return
        payload = gzip.compress(b"OK") if self.path == "/gzip" else b"OK"
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Set-Cookie", "private=value")
        if self.path == "/gzip":
            self.send_header("Content-Encoding", "gzip")
        if self.path == "/close":
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()

    def log_message(self, *args):
        pass


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "*")
    API.seen = []
    API.cookies = []
    transport.close_transport()
    server = ThreadingHTTPServer(("127.0.0.1", 0), API)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    transport.close_transport()
    server.shutdown()
    server.server_close()
    thread.join(2)


@pytest.fixture
def traces(monkeypatch):
    result = []
    monkeypatch.setattr(transport, "queue_http_timing", lambda metadata, events: result.append((metadata, events)))
    return result


def url(server, path="/ok", scheme="http"):
    return f"{scheme}://127.0.0.1:{server.server_port}{path}"


def test_warmed_connection_reused_across_threads_and_rotating_keys(server, traces):
    assert transport.request("HEAD", url(server), auth=_no_auth).status_code == 401
    for key in ("dummy-key-one", "dummy-key-two"):
        with ThreadPoolExecutor(max_workers=1) as pool:
            response = pool.submit(transport.request, "POST", url(server), data=b"audio",
                                   headers={"Authorization": f"Bearer {key}"}).result(3)
            assert response.content == b"OK"
    assert len({entry[1] for entry in API.seen}) == 1
    assert [entry[2] for entry in API.seen] == [None, "Bearer dummy-key-one", "Bearer dummy-key-two"]
    assert [metadata["reused"] for metadata, _ in traces] == [False, True, True]
    assert len({metadata["connection"] for metadata, _ in traces}) == 1
    assert traces[-1][0]["upload_bytes"] == 5
    assert "dns_start" not in traces[-1][1]
    assert all(cookie is None for cookie in API.cookies)


def test_first_byte_precedes_headers_and_complete_body(server, traces):
    timing = OperationTiming("recording")
    assert transport.request("POST", url(server, "/slow"), data=b"payload", timing=timing,
                             stage="transcription").content == b"OK"
    metadata, events = traces[0]
    assert metadata["op"] == timing.operation_id
    phases = ["dns_start", "dns_end", "tcp_start", "tcp_end", "headers_sent", "upload_start",
              "request_sent", "first_byte", "headers_received", "body_end"]
    assert [events[name] for name in phases] == sorted(events[name] for name in phases)
    assert 50_000_000 < events["first_byte"] - events["request_sent"]
    assert 50_000_000 < events["headers_received"] - events["first_byte"]
    assert 50_000_000 < events["body_end"] - events["headers_received"]
    assert timing._events["transcription_http_first_byte"] == events["first_byte"]
    assert metadata["response_body_bytes"] == 2


def test_concurrent_calls_keep_credentials_and_operations_isolated(server, traces):
    def send(index):
        timing = OperationTiming()
        response = transport.request("POST", url(server, "/slow"), data=str(index).encode(), timing=timing,
                                     headers={"Authorization": f"Bearer key-{index}"})
        assert response.content == b"OK"
        return timing.operation_id

    with ThreadPoolExecutor(max_workers=12) as pool:
        ids = list(pool.map(send, range(12)))
    assert len(set(ids)) == 12
    assert {metadata["op"] for metadata, _ in traces} == set(ids)
    assert all(entry[2] == f"Bearer key-{entry[3].decode()}" for entry in API.seen)
    assert all(events["request_sent"] <= events["first_byte"] for _, events in traces)


def test_redirects_have_separate_exchanges(server, traces):
    assert transport.request("POST", url(server, "/redirect"), data=b"audio").content == b"OK"
    assert [metadata["exchange"] for metadata, _ in traces] == [1, 2]
    assert [metadata["status"] for metadata, _ in traces] == [307, 200]
    assert len({metadata["op"] for metadata, _ in traces}) == 1
    assert traces[1][0]["reused"] is True


@pytest.mark.parametrize("path", ["/chunked", "/gzip", "/close"])
def test_response_framing_and_decompression_unchanged(server, traces, path):
    assert transport.request("POST", url(server, path), data=b"audio").content == b"OK"
    assert len(traces) == 1
    assert "body_end" in traces[0][1]
    assert traces[0][0]["response_body_bytes"] >= 2
    assert transport.request("POST", url(server), data=b"again").content == b"OK"
    assert traces[1][0]["reused"] is (path != "/close")


def test_failed_post_is_not_retried_and_has_partial_timing(server, traces):
    with pytest.raises(requests.ConnectionError):
        transport.request("POST", url(server, "/drop"), data=b"audio")
    assert len(API.seen) == 1
    assert len(traces) == 1
    metadata, events = traces[0]
    assert metadata["error"] == "ConnectionError"
    assert metadata["last_phase"] == "request_sent"
    assert "request_sent" in events
    assert "first_byte" not in events
    assert "body_end" not in events


def test_read_timeout_logs_failure_and_releases_connection(server, traces):
    with pytest.raises(requests.ReadTimeout):
        transport.request("POST", url(server, "/slow"), timeout=(1, 0.02))
    assert traces[0][0]["error"] == "ReadTimeout"
    assert "first_byte" not in traces[0][1]
    assert transport.request("POST", url(server), data=b"audio").content == b"OK"


def test_tls_verification_and_reuse(server, traces):
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(CERT, KEY)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    endpoint = url(server, scheme="https")
    with pytest.raises(requests.exceptions.SSLError):
        transport.request("POST", endpoint, data=b"audio")
    assert traces[0][0]["error"] == "SSLError"
    assert "tls_start" in traces[0][1] and "tls_end" not in traces[0][1]
    for _ in range(2):
        assert transport.request("POST", endpoint, data=b"audio", verify=str(CERT)).content == b"OK"
    assert "tls_end" in traces[1][1]
    assert "tls_start" not in traces[2][1]
    assert traces[2][0]["reused"] is True


class Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_CONNECT(self):
        host, port = self.path.split(":")
        with socket.create_connection((host, int(port)), timeout=2) as upstream:
            time.sleep(0.06)
            self.send_response(200)
            self.end_headers()
            sockets = (self.connection, upstream)
            while True:
                ready, _, _ = select.select(sockets, (), (), 2)
                if not ready:
                    return
                for source in ready:
                    data = source.recv(65536)
                    if not data:
                        return
                    target = upstream if source is self.connection else self.connection
                    target.sendall(data)

    def do_POST(self):
        parts = urlsplit(self.path)
        data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=2)
        try:
            conn.request("POST", parts.path, data)
            response = conn.getresponse()
            content = response.read()
            self.send_response(response.status)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        finally:
            conn.close()

    def log_message(self, *args):
        pass


@pytest.mark.parametrize("scheme", ["http", "https"])
@pytest.mark.parametrize("proxy_scheme", ["http", "https"])
def test_proxy_routing_and_connect_not_counted_as_api_first_byte(server, traces, scheme, proxy_scheme):
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
    if proxy_scheme == "https":
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(CERT, KEY)
        proxy.socket = context.wrap_socket(proxy.socket, server_side=True)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    if scheme == "https":
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(CERT, KEY)
        server.socket = context.wrap_socket(server.socket, server_side=True)
    try:
        route = f"{proxy_scheme}://127.0.0.1:{proxy.server_port}"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert transport.request("POST", url(server, scheme=scheme), data=b"audio", proxies={scheme: route},
                                     verify=str(CERT)).content == b"OK"
        # urllib3 flags a plain HTTP origin forwarded through an HTTPS proxy as unverified.
        insecure = [warning for warning in caught if issubclass(warning.category, InsecureRequestWarning)]
        assert bool(insecure) is (proxy_scheme == "https" and scheme == "http")
        metadata, events = traces[0]
        assert metadata["route"] == "proxy"
        if scheme == "https":
            assert events["proxy_tunnel_start"] < events["proxy_tunnel_end"] <= events["tls_start"]
            assert events["tls_end"] <= events["request_sent"] <= events["first_byte"]
            if proxy_scheme == "https":
                assert events["proxy_tls_start"] < events["proxy_tls_end"] <= events["proxy_tunnel_start"]
        elif proxy_scheme == "https":
            assert events["proxy_tls_start"] < events["proxy_tls_end"]
            assert "tls_start" not in events
    finally:
        transport.close_transport()
        proxy.shutdown()
        proxy.server_close()


def test_transport_summary_is_async_and_contains_no_credentials_or_payload(server, caplog):
    with caplog.at_level(logging.INFO):
        transport.request("POST", url(server) + "?private-query", data=b"private-transcript",
                          headers={"Authorization": "Bearer private-key"})
        assert flush_timing_logs()
    line = next(line for line in caplog.messages if line.startswith("http_transport"))
    assert "ttfb_ms=" in line and "after_upload_wait_ms=" in line and "upload_ms=" in line
    assert "private-" not in line and "Authorization" not in line


@pytest.mark.parametrize("endpoint, expected", [
    ("https://api.groq.com/openai/v1/audio/transcriptions", "https://api.groq.com/openai/v1/models"),
    ("https://api.openai.com/v1/chat/completions", "https://api.openai.com/v1/models"),
    ("https://user:password@custom.test:443/infer?secret", "https://custom.test:443/"),
    ("http://[::1]:1234/infer", "http://[::1]:1234/"),
    ("file:///secret", ""),
])
def test_warm_targets_strip_secrets_and_avoid_inference(endpoint, expected):
    assert warm_url(endpoint) == expected


def test_warmup_is_throttled_auth_free_and_never_waits_for_network(server, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []
    checked = threading.Event()
    resolutions = []

    def resolve(*args):
        resolutions.append(args)
        if len(resolutions) == 2:
            checked.set()
        return None

    def warm_request(*args, **kwargs):
        calls.append((args, kwargs))
        entered.set()
        assert release.wait(3)

    monkeypatch.setattr("app.services.http_warmup.request", warm_request)
    monkeypatch.setattr("app.services.http_warmup.resolve_proxies", resolve)
    warm = HttpWarmup()
    try:
        warm.schedule((url(server), url(server, "/chat")), "", False)
        assert entered.wait(1)
        completed = threading.Event()
        threading.Thread(target=lambda: (warm.schedule((url(server),), "", False), completed.set()), daemon=True).start()
        assert completed.wait(0.5)
        release.set()
        assert checked.wait(1)
        warm.close()
        warm._thread.join(2)
        assert len(calls) == 1
        args, kwargs = calls[0]
        assert args == ("HEAD", url(server, "/"))
        assert kwargs["auth"] is _no_auth and kwargs["allow_redirects"] is False
        assert "headers" not in kwargs
    finally:
        release.set()
        warm.close()


def test_warm_request_disables_netrc_credentials(server, monkeypatch):
    monkeypatch.setattr(requests.sessions, "get_netrc_auth", lambda *_: ("user", "password"))
    transport.request("HEAD", url(server), auth=_no_auth)
    assert API.seen[0][2] is None


def test_dns_failure_logs_the_phase_without_fake_tcp_timing(server, traces, monkeypatch):
    def fail(*_args, **_kwargs):
        raise socket.gaierror("synthetic DNS failure")

    monkeypatch.setattr(transport.socket, "getaddrinfo", fail)
    with pytest.raises(requests.ConnectionError):
        transport.request("POST", url(server), data=b"audio")
    metadata, events = traces[0]
    assert metadata["last_phase"] == "dns_start"
    assert "dns_end" not in events and "tcp_start" not in events
