"""Keep the API origins' connections warm while the user is active."""
from __future__ import annotations

from typing import Any, Callable, Dict

from PyQt6.QtCore import QObject, QTimer

from app.services.http_warmup import WARM_INTERVAL_S, HttpWarmup


class WarmupScheduler(QObject):
    """Re-warm the configured endpoints periodically and right after user activity."""

    def __init__(self, config: Dict[str, Any], is_recording: Callable[[], bool],
                 rephrasing_first: Callable[[], bool]) -> None:
        """``rephrasing_first`` puts the rephrasing origin first when a prompt was chosen."""
        super().__init__()
        self._config = config
        self._is_recording = is_recording
        self._rephrasing_first = rephrasing_first
        self.http = HttpWarmup()
        self._timer = QTimer(self)
        self._timer.setInterval(int(WARM_INTERVAL_S * 1000))
        self._timer.timeout.connect(self.schedule)
        self._timer.start()
        QTimer.singleShot(0, self.schedule)

    def touch(self) -> None:
        """Extend the warm window without probing (a request is about to use the pool)."""
        self.http.touch()

    def schedule(self, *, activate: bool = False) -> None:
        """Snapshot endpoints only; keep active origins warm for five minutes after use."""
        if activate or self._is_recording():
            self.http.touch()
        if not self.http.active:
            return
        config = self._config
        endpoints = (config.get("api_endpoint", ""), config.get("rephrasing_api_url", ""))
        if self._rephrasing_first():
            endpoints = endpoints[::-1]
        self.http.schedule(endpoints, config.get("proxy_url", ""), config.get("use_local_px_proxy", False))

    def close(self) -> None:
        """Stop warming at quit."""
        self._timer.stop()
        self.http.close()
