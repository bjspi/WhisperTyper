"""Which rephrasing models get a temperature (sending one to a fixed-temperature model fails with HTTP 400)."""
from __future__ import annotations

import pytest

from app.core.models import rephrasing_supports_temperature


@pytest.mark.parametrize("model", [
    "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5-mini-2025-08-07", "gpt-5.1",
    "gpt-5.6-luna", "gpt-5.6-sol", "gpt-6-luna", "gpt-6.1-sol",
])
def test_reasoning_models_keep_their_default_temperature(model):
    assert not rephrasing_supports_temperature(model)


@pytest.mark.parametrize("model", [
    "gpt-5.2", "gpt-5.4", "gpt-5.4-mini", "gpt-5.5", "gpt-4.1-mini", "gpt-4o", "llama-3.3-70b-versatile",
])
def test_other_models_accept_a_temperature(model):
    assert rephrasing_supports_temperature(model)
