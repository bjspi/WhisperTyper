"""Tests for app/core/textutil.py."""
from __future__ import annotations

from app.core.textutil import clean_model_name, demojibake, estimate_tokens, shorten


class TestDemojibake:
    def test_repairs_latin1_mojibake(self):
        assert demojibake("fÃ¼hrt") == "führt"

    def test_repairs_cp1252_mojibake(self):
        assert demojibake("Ãœbersetzer") == "Übersetzer"

    def test_clean_text_is_untouched(self):
        assert demojibake("Über die Brücke — no markers here") == "Über die Brücke — no markers here"

    def test_non_string_passes_through(self):
        assert demojibake(42) == 42
        assert demojibake(None) is None
        assert demojibake(["Ã¼"]) == ["Ã¼"]


class TestEstimateTokens:

    def test_words_and_punctuation(self):
        assert estimate_tokens("Hello, world!") == 4  # hello , world !


class TestCleanModelName:
    def test_strips_provider_annotation(self):
        assert clean_model_name("whisper-1 (openai)") == "whisper-1"
        assert clean_model_name("whisper-large-v3-turbo (groq)") == "whisper-large-v3-turbo"


class TestShorten:

    def test_whitespace_collapsed(self):
        assert shorten("hello\n\n   world\t!") == "hello world !"

    def test_long_text_clipped_with_ellipsis(self):
        text = "x" * 300
        result = shorten(text, limit=100)
        assert result.endswith(" […]")
        assert len(result) <= 100 + len(" […]")
