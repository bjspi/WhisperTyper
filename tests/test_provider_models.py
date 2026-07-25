"""Tests for provider model catalog fetching."""
from __future__ import annotations

from typing import Any

from app.services.provider_models import fetch_provider_catalog


def test_catalog_is_fetched_once_and_split_by_feature(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            return {
                "data": [
                    {"id": "whisper", "input_modalities": ["audio"], "output_modalities": ["text"]},
                    {"id": "chat", "input_modalities": ["text"], "output_modalities": ["text"]},
                ]
            }

    def fake_get(url: str, **kwargs: Any) -> Response:
        calls.append({"url": url, **kwargs})
        return Response()

    monkeypatch.setattr("app.services.provider_models.requests.get", fake_get)
    catalog = fetch_provider_catalog("groq", "gsk-secret")
    assert catalog == {"transcription": ["whisper"], "rephrasing": ["chat"]}
    assert len(calls) == 1
    assert calls[0]["url"].endswith("/models")
    assert calls[0]["headers"]["Authorization"] == "Bearer gsk-secret"
