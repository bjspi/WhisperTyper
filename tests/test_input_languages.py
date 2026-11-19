"""Spoken-language choices in the recording palette and their config defaults."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.constants import DEFAULT_CONFIG
from app.core.input_languages import DETECT_LANGUAGE, MAX_PALETTE_LANGUAGES, default_palette_languages, palette_languages


def test_choices_follow_catalog_order_and_drop_duplicate_or_unknown_codes():
    configured = ["de", "unknown", "de", "en", DETECT_LANGUAGE, "EN", 3, None, ["fr"], {"es": True}]
    assert palette_languages(configured, "it") == [DETECT_LANGUAGE, "en", "de", "it"]


@pytest.mark.parametrize("configured", [None, "en", 5, ("en",), {"en": True}])
def test_nonlist_config_leaves_only_the_current_language(configured):
    assert palette_languages(configured, "de") == ["de"]


def test_detect_is_a_valid_current_language_even_with_no_configured_choices():
    assert palette_languages([], DETECT_LANGUAGE) == [DETECT_LANGUAGE]


def test_unknown_current_language_is_excluded():
    assert palette_languages(["de"], "unknown") == ["de"]


@pytest.mark.parametrize(
    "configured",
    [
        ["cy", "it", "es", "fr", "de", "en", DETECT_LANGUAGE],
        ["it", "es", "fr", "de", "en", DETECT_LANGUAGE],
    ],
)
def test_current_language_is_kept_at_the_limit_even_when_it_is_last(configured):
    assert palette_languages(configured, "cy") == [DETECT_LANGUAGE, "en", "de", "fr", "cy"]


def test_a_current_language_already_within_the_limit_keeps_its_slot():
    languages = [DETECT_LANGUAGE, "en", "de", "fr", "es", "it", "nl"]
    assert palette_languages(languages, "en") == languages[:MAX_PALETTE_LANGUAGES]


@pytest.mark.parametrize(
    "language, expected",
    [
        ("de", [DETECT_LANGUAGE, "de"]),
        ("en", [DETECT_LANGUAGE, "en"]),
        (DETECT_LANGUAGE, [DETECT_LANGUAGE]),
    ],
)
def test_defaults_offer_detection_and_the_initial_language(language, expected):
    assert default_palette_languages(language) == expected


def test_code_and_example_use_the_same_defaults_for_their_initial_languages():
    example = json.loads((Path(__file__).resolve().parents[1] / "config.example.json").read_text(encoding="utf-8"))
    for config in (DEFAULT_CONFIG, example):
        assert config["recording_palette_language_switch"] is False
        assert config["recording_palette_languages"] == default_palette_languages(config["input_language"])
