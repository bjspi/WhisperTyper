"""Tests for proxy resolution and the connectivity helpers in app/core/netutil.py + services/net.py."""
from __future__ import annotations

import io
import wave

from app.core import netutil
from app.services import net


class TestBuildProxies:
    def test_empty_returns_none(self):
        assert netutil.build_proxies("") is None
        assert netutil.build_proxies("   ") is None
        assert netutil.build_proxies(None) is None

    def test_url_maps_to_both_schemes(self):
        proxies = netutil.build_proxies("http://proxy:8080")
        assert proxies == {"http": "http://proxy:8080", "https": "http://proxy:8080"}


class TestResolveProxies:
    def test_explicit_proxy_wins(self):
        proxies = net.resolve_proxies("http://corp-proxy:3128", use_px=False)
        assert proxies == {"http": "http://corp-proxy:3128", "https": "http://corp-proxy:3128"}

    def test_no_proxy_and_no_px_returns_none(self):
        assert net.resolve_proxies("", use_px=False) is None


class TestGenerateTestWav:
    def test_produces_valid_nonempty_wav(self):
        data = netutil.generate_test_wav_bytes()
        with wave.open(io.BytesIO(data), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() > 0


class TestTcpCheck:
    def test_none_host_is_false(self):
        assert netutil.tcp_check(None, 443) is False
