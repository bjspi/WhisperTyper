"""Configuration persistence: load (with migrations) and save the JSON config.

Single responsibility: own the on-disk config format and the one-time migrations that keep
older config files working. No Qt, no app state — the caller passes in a hotkey normalizer
(so this module stays independent of the hotkey mixin) and receives a plain dict back.
"""
from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Any, Callable, Dict, Tuple

from app.core.constants import (
    CONFIG_SCHEMA_VERSION,
    DEFAULT_CONFIG,
    DEFAULT_REPHRASING_MODEL,
    DEFAULT_TRANSCRIPTION_MODEL,
    LANGUAGES,
    PREVIOUS_DEFAULT_REPHRASING_MODELS,
    WINDOW_MIN_HEIGHT,
)
from app.core.providers import (
    FALLBACK_MODELS,
    clean_legacy_model_name,
    infer_provider,
)
from app.core.textutil import demojibake

# A hotkey normalizer, e.g. HotkeyMixin.normalize_hotkey_string.
HotkeyNormalizer = Callable[[str], str]


class ConfigStore:
    """Reads/writes the JSON config file and applies backward-compatible migrations."""

    def __init__(self, config_file: str, normalize_hotkey: HotkeyNormalizer) -> None:
        """
        Args:
            config_file: Absolute path to the JSON config file.
            normalize_hotkey: Callable that canonicalizes a hotkey string (kept external so
                this module does not depend on the hotkey mixin).
        """
        self.config_file = config_file
        self._normalize_hotkey = normalize_hotkey

    def load(self) -> Tuple[Dict[str, Any], bool]:
        """Load the config, applying migrations and filling in defaults.

        Returns:
            (config, changed): the resolved config dict and whether anything was migrated or
            defaulted (in which case the caller should persist it back via ``save``).
        """
        loaded_config: Dict[str, Any] = {}
        try:
            # Match save()'s utf-8 encoding so non-ASCII prompts (e.g. German umlauts) load
            # correctly on platforms whose default encoding is not utf-8.
            with open(self.config_file, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # File doesn't exist or is corrupted, will proceed with defaults
            pass

        changed = self._migrate(loaded_config)
        return loaded_config, changed

    def save(self, config: Dict[str, Any]) -> None:
        """Write the current configuration to the JSON file."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                # ensure_ascii=False keeps the file human-readable UTF-8 (ü, é, …) and,
                # paired with the utf-8 read above, avoids the classic mojibake round-trip.
                json.dump(config, f, indent=4, ensure_ascii=False)
            logging.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logging.error(f"Failed to save configuration: {e}")

    def _migrate(self, cfg: Dict[str, Any]) -> bool:
        """Apply all in-place migrations/defaults to ``cfg``; return True if it changed."""
        changed = False

        # Schema v2 stores known-provider credentials once and lets both features select one.
        # Legacy custom endpoints remain separate because they may genuinely be different APIs.
        legacy_provider_keys = {"api_key", "api_endpoint", "rephrasing_api_url", "rephrasing_api_key"}
        if cfg.get("config_schema_version", 0) < 2 or legacy_provider_keys & cfg.keys():
            keys = deepcopy(cfg.get("provider_api_keys", {}))
            if not isinstance(keys, dict):
                keys = {}
            keys.setdefault("openai", "")
            keys.setdefault("groq", "")
            custom = deepcopy(cfg.get("custom_provider_settings", {}))
            if not isinstance(custom, dict):
                custom = {}
            for feature in ("transcription", "rephrasing"):
                if not isinstance(custom.get(feature), dict):
                    custom[feature] = {"endpoint": "", "api_key": ""}
                custom[feature].setdefault("endpoint", "")
                custom[feature].setdefault("api_key", "")

            transcription_endpoint = str(
                cfg.get("api_endpoint", "https://api.openai.com/v1/audio/transcriptions") or ""
            )
            transcription_provider = str(
                cfg.get("transcription_provider") or infer_provider(transcription_endpoint)
            )
            transcription_key = str(cfg.get("api_key", "") or "")
            if transcription_provider in keys:
                if transcription_key and not keys[transcription_provider]:
                    keys[transcription_provider] = transcription_key
            else:
                custom["transcription"] = {
                    "endpoint": transcription_endpoint,
                    "api_key": transcription_key,
                }

            rephrase_endpoint = str(
                cfg.get("rephrasing_api_url", "https://api.openai.com/v1/chat/completions") or ""
            )
            rephrase_key = str(cfg.get("rephrasing_api_key", "") or "")
            rephrase_provider = str(cfg.get("rephrasing_provider") or infer_provider(rephrase_endpoint))

            # Old configs commonly left the secondary defaults untouched. If transcription is
            # already configured, share that provider/key instead of manufacturing a second setup.
            untouched_secondary = (
                rephrase_endpoint == "https://api.openai.com/v1/chat/completions"
                and not rephrase_key
                and transcription_provider in ("openai", "groq")
                and bool(keys.get(transcription_provider))
            )
            if untouched_secondary:
                rephrase_provider = transcription_provider
            if rephrase_provider in keys:
                if rephrase_key and not keys[rephrase_provider]:
                    keys[rephrase_provider] = rephrase_key
            else:
                custom["rephrasing"] = {
                    "endpoint": rephrase_endpoint,
                    "api_key": rephrase_key,
                }

            cfg["provider_api_keys"] = keys
            cfg["custom_provider_settings"] = custom
            cfg["transcription_provider"] = transcription_provider
            cfg["rephrasing_provider"] = rephrase_provider
            for key in legacy_provider_keys:
                cfg.pop(key, None)

            cfg["model"] = clean_legacy_model_name(
                str(cfg.get("model", DEFAULT_TRANSCRIPTION_MODEL))
            )
            old_rephrase_model = str(cfg.get("rephrasing_model", DEFAULT_REPHRASING_MODEL))
            if (
                rephrase_provider == "groq"
                and old_rephrase_model in PREVIOUS_DEFAULT_REPHRASING_MODELS | {DEFAULT_REPHRASING_MODEL}
            ):
                cfg["rephrasing_model"] = FALLBACK_MODELS["groq"]["rephrasing"][0]
            else:
                cfg["rephrasing_model"] = clean_legacy_model_name(old_rephrase_model)
            changed = True

        # Migrate old 'language' key to 'input_language'
        if "language" in cfg and "input_language" not in cfg:
            cfg["input_language"] = cfg.pop("language")
            changed = True

        # Check if input_language is a display name and convert to code
        if "input_language" in cfg:
            lang_value = cfg["input_language"]
            # If it's a name (e.g., "German", length > 2), convert it to code
            if isinstance(lang_value, str) and len(lang_value) > 2 and lang_value in LANGUAGES:
                cfg["input_language"] = LANGUAGES[lang_value]
                changed = True

        # Migrate the rephrasing model from a previous built-in default to the current one,
        # but only if the user never chose their own model (still on an old default).
        if cfg.get("rephrasing_model") in PREVIOUS_DEFAULT_REPHRASING_MODELS:
            cfg["rephrasing_model"] = DEFAULT_REPHRASING_MODEL
            changed = True

        # Self-heal mojibake: UTF-8 text once mis-decoded as latin-1 ("fÃ¼hrt" -> "führt").
        # Repairs text fields (incl. rephrase entries) on load, regardless of disk state.
        for cfg_key, cfg_value in list(cfg.items()):
            repaired = demojibake(cfg_value)
            if repaired != cfg_value:
                cfg[cfg_key] = repaired
                changed = True
        for entry in cfg.get("post_rephrasing_entries", []):
            if isinstance(entry, dict):
                for sub_key in ("caption", "text"):
                    repaired = demojibake(entry.get(sub_key, ""))
                    if repaired != entry.get(sub_key, ""):
                        entry[sub_key] = repaired
                        changed = True

        # Schema migration: configs from before the redesign (no/older schema version) get their
        # window height bumped to at least the minimum that fits the new UI, once.
        if cfg.get("config_schema_version", 0) < CONFIG_SCHEMA_VERSION:
            current_h = int(cfg.get("window_height", 0) or 0)
            cfg["window_height"] = max(current_h, WINDOW_MIN_HEIGHT)
            cfg["config_schema_version"] = CONFIG_SCHEMA_VERSION
            changed = True

        # Ensure all default keys exist in the loaded config
        for key, default_value in DEFAULT_CONFIG.items():
            if key not in cfg:
                cfg[key] = deepcopy(default_value)
                changed = True

        # Self-heal hotkeys polluted by a captured control char. If "Set hotkey" was active while
        # the key's own global action fired, its simulated Ctrl+C (\x03) got captured too, saving
        # garbage like "<ctrl>+\x03+<f9>+c" (shown as "<ctrl>++<f9>+c") that no longer binds.
        for hk_key in ("hotkey", "post_rephrase_hotkey"):
            hk_val = cfg.get(hk_key, "")
            if isinstance(hk_val, str) and any(ord(ch) < 32 for ch in hk_val):
                cfg[hk_key] = DEFAULT_CONFIG.get(hk_key, "")
                changed = True

        for hk_key in ("hotkey", "post_rephrase_hotkey"):
            hk_val = cfg.get(hk_key, "")
            normalized = self._normalize_hotkey(hk_val)
            if hk_val and normalized and normalized != hk_val:
                cfg[hk_key] = normalized
                changed = True

        return changed
