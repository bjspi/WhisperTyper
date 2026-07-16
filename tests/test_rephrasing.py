"""Tests for the provider-compatible rephrasing request builder."""
from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from app.services.rephrasing import RephrasingError, rephrase_text


@pytest.mark.parametrize("model", ["gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"])
def test_gpt_5_6_omits_unsupported_temperature(model: str) -> None:
    response = Mock()
    response.json.return_value = {"choices": [{"message": {"content": "Success"}}]}

    with patch("app.services.rephrasing.requests.post", return_value=response) as post:
        result = rephrase_text(
            system_prompt="Test",
            user_prompt="Reply",
            api_url="https://api.openai.com/v1/chat/completions",
            api_key="test-key",
            model=model,
            temperature=0.0,
        )

    assert result == "Success"
    assert "temperature" not in post.call_args.kwargs["json"]


def test_other_models_keep_configured_temperature() -> None:
    response = Mock()
    response.json.return_value = {"choices": [{"message": {"content": "Success"}}]}

    with patch("app.services.rephrasing.requests.post", return_value=response) as post:
        rephrase_text(
            system_prompt="Test",
            user_prompt="Reply",
            api_url="https://example.com/v1/chat/completions",
            api_key="test-key",
            model="gpt-4o-mini",
            temperature=0.7,
        )

    assert post.call_args.kwargs["json"]["temperature"] == 0.7


def test_api_error_exposes_structured_diagnostics() -> None:
    response = Mock()
    response.ok = False
    response.status_code = 400
    response.headers = {"x-request-id": "req_test_123"}
    response.json.return_value = {
        "error": {
            "message": "Unsupported value: 'temperature' does not support 0 with this model.",
            "type": "invalid_request_error",
            "param": "temperature",
            "code": "unsupported_value",
        }
    }

    with patch("app.services.rephrasing.requests.post", return_value=response):
        with pytest.raises(RephrasingError) as exc_info:
            rephrase_text(
                system_prompt="Test",
                user_prompt="Reply",
                api_url="https://api.openai.com/v1/chat/completions",
                api_key="test-key",
                model="test-model",
                temperature=0.0,
            )

    detail = str(exc_info.value)
    assert "HTTP 400" in detail
    assert "Unsupported value" in detail
    assert "Parameter: temperature" in detail
    assert "Code: unsupported_value" in detail
    assert "Type: invalid_request_error" in detail
    assert "Request ID: req_test_123" in detail


def test_non_json_api_error_uses_bounded_response_text() -> None:
    response = Mock()
    response.ok = False
    response.status_code = 502
    response.headers = {}
    response.text = "Gateway unavailable"
    response.reason = "Bad Gateway"
    response.json.side_effect = ValueError("not JSON")

    with patch("app.services.rephrasing.requests.post", return_value=response):
        with pytest.raises(RephrasingError) as exc_info:
            rephrase_text(
                system_prompt="Test",
                user_prompt="Reply",
                api_url="https://example.com/v1/chat/completions",
                api_key="test-key",
                model="test-model",
                temperature=0.7,
            )

    detail = str(exc_info.value)
    assert "HTTP 502" in detail
    assert "Gateway unavailable" in detail
