"""Proxy / connectivity primitives. Single responsibility: outbound-networking helpers.

Pure functions — no Qt, no app state. Used by the transcription/rephrasing services and
the settings connection tests.
"""

from __future__ import annotations

import io
import math
import socket
import struct
import wave
from typing import Dict, Optional

# "px" (https://github.com/genotrance/px) is a popular local proxy that transparently handles
# corporate NTLM/Kerberos authentication and listens on 127.0.0.1:3128 by default. If it is
# running, pointing WhisperTyper at it lets requests get out through the corporate proxy.
PX_PROXY_HOST = "127.0.0.1"
PX_PROXY_PORT = 3128
PX_PROXY_URL = f"http://{PX_PROXY_HOST}:{PX_PROXY_PORT}"


def tcp_check(host: Optional[str], port: int, timeout: float = 3.0) -> bool:
    """Return True if a raw TCP connection to host:port can be established (best-effort)."""
    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def is_px_running(timeout: float = 1.0) -> bool:
    """Return True if a local proxy (px) appears to be listening on 127.0.0.1:3128."""
    return tcp_check(PX_PROXY_HOST, PX_PROXY_PORT, timeout=timeout)


def build_proxies(proxy_url: str) -> Optional[Dict[str, str]]:
    """
    Build a ``requests`` proxies dict from a configured proxy URL.

    Empty ``proxy_url`` returns None so ``requests`` falls back to system/environment
    proxy settings.
    """
    proxy_url = (proxy_url or "").strip()
    if not proxy_url:
        return None
    return {"http": proxy_url, "https": proxy_url}


def generate_test_wav_bytes(duration_s: float = 0.4, samplerate: int = 16000) -> bytes:
    """Generate a tiny, valid (near-silent) mono 16-bit WAV file in memory for connection tests."""
    num_frames = int(duration_s * samplerate)
    # Very low-amplitude tone so the file is a valid, non-empty audio payload.
    frames = bytearray()
    for i in range(num_frames):
        sample = int(30 * math.sin(2 * math.pi * 220 * (i / samplerate)))
        frames += struct.pack("<h", sample)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(bytes(frames))
    return buffer.getvalue()
