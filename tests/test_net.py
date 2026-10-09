"""Tests for proxy resolution and the connectivity helpers in app/services/netutil.py + services/net.py."""
from __future__ import annotations

import io
import wave

from app.services import net, netutil


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

    def test_config_snapshot_uses_its_proxy_fields(self, monkeypatch):
        monkeypatch.setattr(net, "is_px_running", lambda: True)
        assert net.proxies_for({"proxy_url": " http://corp:3128 ", "use_local_px_proxy": True})["https"] == "http://corp:3128"
        assert net.proxies_for({"proxy_url": "", "use_local_px_proxy": True})["https"] == net.build_proxies(net.PX_PROXY_URL)["https"]
        assert net.proxies_for({}) is None


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
