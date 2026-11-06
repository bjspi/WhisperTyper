"""The first-start guide: its answers set up the configuration by the same rules as the settings."""
from __future__ import annotations

import copy
import json
import os
from unittest.mock import Mock

import pytest

from app.core.api_keys import rephrasing_configured, resolve_key_profile, transcription_configured
from app.core.constants import DEFAULT_CONFIG
from app.core.onboarding import ApiAccess, OnboardingChoices, apply_languages, apply_onboarding
from app.core.prompts import (
    DEFAULT_ENGLISH_PROMPTS,
    DEFAULT_INSTRUCTION_CAPTIONS,
    DEFAULT_LIVEPROMPT_SYSTEM_PROMPTS,
    DEFAULT_TRANSCRIPTION_PROMPTS,
    first_start_transformations,
)


@pytest.fixture
def config() -> dict:
    """A new installation started in English."""
    fresh = copy.deepcopy(DEFAULT_CONFIG)
    fresh.update(api_key_profiles=[], ui_language="en", input_language="en",
                 prompt=DEFAULT_TRANSCRIPTION_PROMPTS["en"].strip(), rephrasing_enabled=False, onboarding_done=False,
                 post_rephrasing_entries=first_start_transformations("en"))
    return fresh


def test_german_languages_also_set_the_prompts_to_german(config):
    apply_languages(config, "de", "de")
    assert config["ui_language"] == config["input_language"] == "de"
    assert config["prompt"] == DEFAULT_TRANSCRIPTION_PROMPTS["de"].strip()
    instruction, english = config["post_rephrasing_entries"]
    assert instruction["caption"] == DEFAULT_INSTRUCTION_CAPTIONS["de"]
    assert instruction["text"] == DEFAULT_LIVEPROMPT_SYSTEM_PROMPTS["de"].strip()
    assert english["text"] == DEFAULT_ENGLISH_PROMPTS["de"].strip()


def test_the_transcription_prompt_follows_the_spoken_language(config):
    apply_languages(config, "de", "fr")
    assert config["prompt"] == DEFAULT_TRANSCRIPTION_PROMPTS["fr"].strip()


def test_an_edited_prompt_stays(config):
    config["prompt"] = "My own words"
    apply_languages(config, "de", "de")
    assert config["prompt"] == "My own words"


def test_api_transcription_with_rephrasing_switched_off(config):
    apply_onboarding(config, OnboardingChoices("de", "de", ApiAccess("groq", " gsk_abc "), None))
    assert config["api_endpoint"] == "https://api.groq.com/openai/v1/audio/transcriptions"
    assert config["model"] == "whisper-large-v3-turbo"
    assert config["api_key_profiles"][0]["key"] == "gsk_abc"
    assert transcription_configured(config) and not rephrasing_configured(config)
    assert not config["local_engine_setup_pending"]


def test_one_key_for_both_tasks_is_stored_once(config):
    access = ApiAccess("openai", "sk-abc")
    apply_onboarding(config, OnboardingChoices("en", "en", access, access))
    assert [profile["key"] for profile in config["api_key_profiles"]] == ["sk-abc"]
    assert transcription_configured(config) and rephrasing_configured(config)
    assert resolve_key_profile(config, "rephrasing")["key"] == "sk-abc"
    assert config["model"] == DEFAULT_CONFIG["model"]
    assert config["rephrasing_model"] == DEFAULT_CONFIG["rephrasing_model"]


def test_local_transcription_stays_pending_for_the_background_setup(config):
    apply_onboarding(config, OnboardingChoices("de", "de", None, ApiAccess("groq", "gsk_abc")))
    assert config["api_endpoint"].startswith("local://")
    assert config["local_engine_enabled"] and config["local_engine_setup_pending"]
    assert "groq" in config["rephrasing_api_url"] and config["rephrasing_enabled"]


def test_a_missing_key_adds_no_profile_and_settings_stay_incomplete(config):
    apply_onboarding(config, OnboardingChoices("en", "en", ApiAccess("openai"), None))
    assert config["api_key_profiles"] == [] and not transcription_configured(config)


# --- The dialog -----------------------------------------------------------------------------
QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.audio.store import RecordingStore  # noqa: E402
from app.context import AppContext  # noqa: E402
from app.core.config_store import ConfigStore  # noqa: E402
from app.core.hotkeys import normalize_hotkey_string  # noqa: E402
from app.ui import onboarding as onboarding_ui  # noqa: E402
from app.ui.onboarding import OnboardingDialog  # noqa: E402

QAPP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])  # must stay referenced for the whole run


@pytest.fixture
def ctx(tmp_path) -> AppContext:
    """A new installation on disk: no config file yet."""
    context = AppContext(ConfigStore(str(tmp_path / "config.json"), normalize_hotkey_string), RecordingStore(str(tmp_path)))
    context.translator.set_language("en")
    context.config.update(ui_language="en", input_language="en", prompt=DEFAULT_TRANSCRIPTION_PROMPTS["en"].strip())
    return context


def test_choosing_the_app_language_switches_the_guide_and_presets_the_spoken_language(ctx):
    dialog = OnboardingDialog(ctx, local_supported=True)
    dialog._ui_language.setCurrentIndex(dialog._ui_language.findData("de"))
    assert dialog.languages() == ("de", "de")
    assert dialog._next.text() == "Weiter"
    dialog._input_language.setCurrentIndex(dialog._input_language.findData("fr"))
    assert dialog.languages() == ("de", "fr")  # A spoken language chosen afterwards stays.


def test_languages_are_shown_by_their_own_names(ctx):
    dialog = OnboardingDialog(ctx, local_supported=True)
    labels = [dialog._input_language.itemText(index) for index in range(dialog._input_language.count())]
    assert {"Deutsch", "Français", "English", "Español"} <= set(labels) and "German" not in labels
    assert all(labels)
    dialog._ui_language.setCurrentIndex(dialog._ui_language.findData("de"))
    assert dialog._input_language.itemText(0) == "Sprache erkennen"
    assert dialog._input_language.itemData(dialog._input_language.findText("Deutsch")) == "de"


def test_the_window_is_only_as_tall_as_the_question_needs(ctx):
    dialog = OnboardingDialog(ctx, local_supported=True)
    first_question = dialog.height()
    dialog._go_to(2)
    assert dialog.height() > first_question
    dialog._go_to(1)
    assert dialog._pages.height() >= dialog._input_language.sizeHint().height()  # a dropdown page keeps its dropdown
    dialog._go_to(0)
    assert dialog.height() == first_question


def test_the_guide_offers_only_what_the_answers_need(ctx):
    dialog = OnboardingDialog(ctx, local_supported=True)
    dialog._use_local.setChecked(True)
    assert dialog._transcription_access.isHidden() and not dialog._local_hint.isHidden()
    dialog._rephrasing_on.setChecked(True)
    assert not dialog._rephrasing_access.isHidden() and dialog._same_key_hint.isHidden()  # no key to share

    dialog._use_api.setChecked(True)
    dialog._transcription_access.key.setText("sk-abc")
    dialog._update_answer_controls()
    assert dialog._rephrasing_access.isHidden() and not dialog._same_key_hint.isHidden()
    assert dialog.choices().rephrasing == ApiAccess("openai", "sk-abc")


def test_macos_does_not_offer_local_transcription(ctx):
    dialog = OnboardingDialog(ctx, local_supported=False)
    assert dialog._use_local.isHidden()


def _run_with(monkeypatch, drive) -> None:
    """Replace the dialog's event loop by ``drive``."""
    monkeypatch.setattr(OnboardingDialog, "exec", drive)
    monkeypatch.setattr(OnboardingDialog, "show", Mock())


def test_finishing_stores_everything_and_is_not_asked_again(ctx, monkeypatch):
    def finish(dialog):
        dialog._ui_language.setCurrentIndex(dialog._ui_language.findData("de"))
        dialog._transcription_access.key.setText("sk-abc")
        for _page in range(dialog._pages.count()):
            dialog._next.click()

    _run_with(monkeypatch, finish)
    onboarding_ui.run_onboarding(ctx, local_supported=True)
    with open(ctx._store.config_file, encoding="utf-8") as saved:
        stored = json.load(saved)
    assert stored["onboarding_done"] is True and stored["ui_language"] == "de"
    assert stored["prompt"] == DEFAULT_TRANSCRIPTION_PROMPTS["de"].strip()
    assert stored["api_key_profiles"][0]["key"] == "sk-abc"
    assert stored["rephrasing_enabled"] is False


def test_skipping_keeps_the_languages_only(ctx, monkeypatch):
    def skip(dialog):
        dialog._ui_language.setCurrentIndex(dialog._ui_language.findData("de"))
        dialog._transcription_access.key.setText("sk-abc")
        dialog._skip.click()

    _run_with(monkeypatch, skip)
    onboarding_ui.run_onboarding(ctx, local_supported=True)
    assert ctx.config["onboarding_done"] is True and ctx.config["ui_language"] == "de"
    assert not ctx.config.get("api_key_profiles")
    assert not ctx.config["local_engine_setup_pending"]
