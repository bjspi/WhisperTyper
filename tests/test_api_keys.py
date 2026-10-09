"""Credential migration, explicit provider scoping and optional Groq round-robin rotation."""
from __future__ import annotations

from copy import deepcopy

import pytest

from app.core.api_keys import GroqKeyRotation, migrate_api_keys, provider_for_url, selected_api_key
from app.core.constants import DEFAULT_CONFIG


@pytest.mark.parametrize("same_key,same_provider,count", [(True, True, 1), (False, True, 2), (True, False, 2)])
def test_legacy_keys_migrate_without_loss_or_cross_provider_sharing(same_key, same_provider, count):
    config = deepcopy(DEFAULT_CONFIG)
    config.update(api_key="sk-test-a", rephrasing_api_key="sk-test-a" if same_key else "sk-test-b")
    if not same_provider:
        config["rephrasing_api_url"] = "https://api.groq.com/openai/v1/chat/completions"
    assert migrate_api_keys(config)
    assert len(config["api_key_profiles"]) == count
    assert selected_api_key(config, "transcription") == "sk-test-a"
    assert selected_api_key(config, "rephrasing") == ("sk-test-a" if same_key else "sk-test-b")
    assert "api_key" not in config and "rephrasing_api_key" not in config
    snapshot = deepcopy(config)
    assert not migrate_api_keys(config)
    assert config == snapshot


def test_explicit_rephrase_selection_does_not_fall_back_to_transcription():
    config = deepcopy(DEFAULT_CONFIG)
    config.update(api_key="sk-test-a", rephrasing_api_key="")
    migrate_api_keys(config)
    assert selected_api_key(config, "transcription") == "sk-test-a"
    assert selected_api_key(config, "rephrasing") == ""


@pytest.mark.parametrize("endpoint", ["https://api.openai.com.evil.test/v1", "https://evil.test/api.openai.com", "https://api.groq.com/v1"])
def test_selected_key_is_not_used_for_a_different_provider(endpoint):
    config = deepcopy(DEFAULT_CONFIG)
    config["api_key"] = "sk-test-a"
    migrate_api_keys(config)
    config["api_endpoint"] = endpoint
    assert selected_api_key(config, "transcription") == ""
    assert provider_for_url("https://API.OPENAI.COM/v1") == "openai"


def test_adding_legacy_key_preserves_existing_profiles_and_explicit_selection():
    config = rotation_config()
    config["api_key"] = "gsk-legacy-new"
    existing = deepcopy(config["api_key_profiles"])
    assert migrate_api_keys(config)
    assert config["api_key_profiles"][:len(existing)] == existing
    assert config["transcription_key_profile_id"] == "b"
    assert config["api_key_profiles"][-1]["key"] == "gsk-legacy-new"


@pytest.mark.parametrize("profiles", [None, {}, [{"id": "secret-value"}]])
def test_malformed_profiles_are_rejected_without_exposing_their_content(profiles):
    config = deepcopy(DEFAULT_CONFIG)
    config["api_key_profiles"] = profiles
    with pytest.raises(ValueError) as error:
        migrate_api_keys(config)
    assert "secret-value" not in str(error.value)


def rotation_config():
    config = deepcopy(DEFAULT_CONFIG)
    config.update(
        api_endpoint="https://api.groq.com/openai/v1/audio/transcriptions",
        rephrasing_api_url="https://api.groq.com/openai/v1/chat/completions",
        transcription_key_profile_id="b", rephrasing_key_profile_id="a", groq_key_rotation=True,
        api_key_profiles=[
            {"id": "a", "name": "First", "provider": "groq", "key": "gsk-test-a"},
            {"id": "b", "name": "Second", "provider": "groq", "key": "gsk-test-b"},
            {"id": "c", "name": "Third", "provider": "groq", "key": "gsk-test-c"},
            {"id": "other", "name": "Other provider", "provider": "openai", "key": "sk-test"},
            {"id": "empty", "name": "Empty", "provider": "groq", "key": ""},
            {"id": "duplicate", "name": "Duplicate", "provider": "groq", "key": "gsk-test-a"},
            {"id": "bad", "name": "Invalid", "provider": "groq", "key": "gsk-\r\ninjected"},
        ],
    )
    return config


def test_rotation_starts_with_selected_key_and_skips_empty_duplicate_and_other_provider_keys():
    config = rotation_config()
    rotation = GroqKeyRotation()
    assert [rotation.next_key(config) for _ in range(5)] == ["gsk-test-b", "gsk-test-c", "gsk-test-a", "gsk-test-b", "gsk-test-c"]
    assert selected_api_key(config, "rephrasing") == "gsk-test-a"
    assert config["transcription_key_profile_id"] == "b"
    assert rotation.last_profile_id == "c"


def test_disabled_rotation_and_other_providers_use_only_the_chosen_key():
    config = rotation_config()
    config["groq_key_rotation"] = False
    rotation = GroqKeyRotation()
    assert [rotation.next_key(config) for _ in range(3)] == ["gsk-test-b"] * 3
    config["groq_key_rotation"] = True
    config["api_endpoint"] = "https://api.openai.com/v1/audio/transcriptions"
    config["transcription_key_profile_id"] = "other"
    assert [rotation.next_key(config) for _ in range(3)] == ["sk-test"] * 3


def test_missing_or_deleted_selection_never_uses_another_key_implicitly():
    config = rotation_config()
    config["transcription_key_profile_id"] = "deleted"
    assert GroqKeyRotation().next_key(config) == ""


def test_pool_or_selection_changes_restart_from_the_current_selected_key():
    config = rotation_config()
    rotation = GroqKeyRotation()
    assert rotation.next_key(config) == "gsk-test-b"
    config["transcription_key_profile_id"] = "a"
    assert rotation.next_key(config) == "gsk-test-a"
    config["api_key_profiles"][1]["key"] = "gsk-test-b-new"
    assert rotation.next_key(config) == "gsk-test-a"
    assert rotation.next_key(config) == "gsk-test-b-new"


def test_single_key_and_surrounding_whitespace_are_safe():
    config = rotation_config()
    config["api_key_profiles"] = [config["api_key_profiles"][1]]
    config["api_key_profiles"][0]["key"] = " \ngsk-test-b\n "
    rotation = GroqKeyRotation()
    assert rotation.next_key(config) == rotation.next_key(config) == "gsk-test-b"
