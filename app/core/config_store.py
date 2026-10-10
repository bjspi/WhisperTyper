"""Configuration persistence: load (with migrations) and save the JSON config.

Single responsibility: own the on-disk config format and the one-time migrations that keep
older config files working. No Qt, no app state — the caller passes in a hotkey normalizer
(so this module stays independent of the hotkey mixin) and receives a plain dict back.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, Tuple

from app.core.api_keys import migrate_api_keys
from app.core.constants import (
    CONFIG_SCHEMA_VERSION,
    DEFAULT_CONFIG,
    LANGUAGES,
    WINDOW_MIN_HEIGHT,
)
from app.core.hotkeys import is_clipboard_shortcut
from app.core.prompts import INSTRUCTION, default_instruction_entry, transformation_entry
from app.core.textutil import demojibake

# A hotkey normalizer, e.g. HotkeyMixin.normalize_hotkey_string.
HotkeyNormalizer = Callable[[str], str]

#: Top-level LivePrompt keys of older configs and the instruction-entry fields they fill.
_LEGACY_LIVEPROMPT_FIELDS = {
    "liveprompt_enabled": "enabled",
    "liveprompt_trigger_words": "trigger_words",
    "liveprompt_trigger_word_scan_depth": "scan_depth",
    "liveprompt_strip_trigger": "strip_trigger",
    "liveprompt_system_prompt": "text",
    "rephrase_use_selection_context": "use_selection_context",
}


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

        # Migrate old 'language' key to 'input_language'
        if "language" in cfg and "input_language" not in cfg:
            cfg["input_language"] = cfg.pop("language")
            changed = True

        # Fast paste applies to Windows and macOS; keep a choice saved under its former name.
        if "windows_fast_paste" in cfg:
            cfg.setdefault("fast_paste", cfg["windows_fast_paste"])
            del cfg["windows_fast_paste"]
            changed = True

        # Automatic rephrasing is a per-prompt "apply automatically" flag; drop the global switch.
        for removed_key in ("generic_rephrase_enabled", "generic_rephrase_prompt"):
            if removed_key in cfg:
                del cfg[removed_key]
                changed = True

        # Check if input_language is a display name and convert to code
        if "input_language" in cfg:
            lang_value = cfg["input_language"]
            # If it's a name (e.g., "German", length > 2), convert it to code
            if isinstance(lang_value, str) and len(lang_value) > 2 and lang_value in LANGUAGES:
                cfg["input_language"] = LANGUAGES[lang_value]
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
                # Transformation templates predate the per-entry recording-palette flag.
                # Existing templates must stay opt-in so an upgrade never changes normal
                # voice-typing behaviour without the user's explicit choice.
                show_during_recording = entry.get("show_during_recording", False)
                if not isinstance(show_during_recording, bool):
                    show_during_recording = False
                if entry.get("show_during_recording") is not show_during_recording:
                    entry["show_during_recording"] = show_during_recording
                    changed = True

        # Schema migration: configs from before the redesign (no/older schema version) get their
        # window height bumped to at least the minimum that fits the new UI, once.
        if cfg.get("config_schema_version", 0) < CONFIG_SCHEMA_VERSION:
            current_h = int(cfg.get("window_height", DEFAULT_CONFIG["window_height"]) or 0)
            cfg["window_height"] = max(current_h, WINDOW_MIN_HEIGHT)
            cfg["config_schema_version"] = CONFIG_SCHEMA_VERSION
            changed = True

        # Ensure all default keys exist in the loaded config
        for key, default_value in DEFAULT_CONFIG.items():
            if key not in cfg:
                cfg[key] = default_value
                changed = True

        changed = migrate_api_keys(cfg) or changed
        changed = self._migrate_instruction_entry(cfg) or changed

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
            # A hand-edited Select all/Copy/Paste hotkey would break the clipboard system-wide.
            if isinstance(cfg.get(hk_key), str) and is_clipboard_shortcut(cfg[hk_key]):
                logging.warning("Hotkey %s collides with a clipboard shortcut; using the default.", hk_key)
                cfg[hk_key] = DEFAULT_CONFIG[hk_key]
                changed = True

        return changed

    @staticmethod
    def _migrate_instruction_entry(cfg: Dict[str, Any]) -> bool:
        """Keep LivePrompt as the instruction entry of the prompt list (added once, at the top).

        Its settings come from the top-level LivePrompt keys of older configs, which are then
        dropped; a fresh config gets the defaults in the UI language.
        """
        changed = False
        entries = cfg.get("post_rephrasing_entries")
        if not isinstance(entries, list):
            entries = []
            changed = True
        if not any(isinstance(entry, dict) and entry.get("kind") == INSTRUCTION for entry in entries):
            instruction = default_instruction_entry(str(cfg.get("ui_language", DEFAULT_CONFIG["ui_language"])))
            for legacy_key, field in _LEGACY_LIVEPROMPT_FIELDS.items():
                if legacy_key in cfg:
                    instruction[field] = cfg[legacy_key]
            entries = [transformation_entry(instruction), *entries]
            changed = True
        cfg["post_rephrasing_entries"] = entries
        for legacy_key in _LEGACY_LIVEPROMPT_FIELDS:
            if legacy_key in cfg:
                del cfg[legacy_key]
                changed = True
        return changed
