"""Tests for the hotkey grammar in app/core/hotkeys.py."""
from __future__ import annotations

from app.core import hotkeys
from app.core.env import is_WINDOWS


class TestNormalizeHotkeyPart:
    def test_function_key_gets_brackets(self):
        assert hotkeys.normalize_hotkey_part("F9") == "<f9>"
        assert hotkeys.normalize_hotkey_part("f12") == "<f12>"

    def test_already_bracketed_token_is_canonicalized(self):
        assert hotkeys.normalize_hotkey_part("<ctrl_l>") == "<ctrl>"

    def test_aliases(self):
        assert hotkeys.normalize_hotkey_part("Control") == "<ctrl>"
        assert hotkeys.normalize_hotkey_part("Command") == "<cmd>"
        assert hotkeys.normalize_hotkey_part("Option") == "<alt>"
        assert hotkeys.normalize_hotkey_part("CapsLock") == "<caps_lock>"
        assert hotkeys.normalize_hotkey_part("Caps Lock") == "<caps_lock>"
        assert hotkeys.normalize_hotkey_part("Windows") == "<win>"
        assert hotkeys.normalize_hotkey_part("PgUp") == "<page_up>"

    def test_empty_input(self):
        assert hotkeys.normalize_hotkey_part("") == ""
        assert hotkeys.normalize_hotkey_part("   ") == ""


class TestNormalizeHotkeyString:
    def test_user_friendly_combo(self):
        assert hotkeys.normalize_hotkey_string("Cmd+Shift+F") == "<cmd>+<shift>+f"

    def test_legacy_format_passes_through(self):
        assert hotkeys.normalize_hotkey_string("<ctrl>+<f9>") == "<ctrl>+<f9>"

    def test_duplicate_tokens_are_dropped(self):
        assert hotkeys.normalize_hotkey_string("ctrl+control+x") == "<ctrl>+x"

    def test_left_right_variants_collapse(self):
        assert hotkeys.normalize_hotkey_string("<caps_lock>+<ctrl_l>") == "<caps_lock>+<ctrl>"

    def test_plus_key_survives_the_separator(self):
        assert hotkeys.normalize_hotkey_string("Ctrl+Plus") == "<ctrl>+<plus>"
        assert hotkeys.normalize_hotkey_string("<ctrl>+<plus>") == "<ctrl>+<plus>"

    def test_typed_plus_key_is_not_lost(self):
        assert hotkeys.normalize_hotkey_string("Ctrl++") == "<ctrl>+<plus>"
        assert hotkeys.normalize_hotkey_string("+") == "<plus>"
        assert hotkeys.normalize_hotkey_string("Ctrl+++Alt") == "<ctrl>+<plus>+<alt>"
        assert hotkeys.normalize_hotkey_string("++Ctrl") == "<plus>+<ctrl>"


class TestFormatHotkeyTokens:
    def test_modifiers_come_first_in_stable_order(self):
        assert hotkeys.format_hotkey_tokens({"x", "<shift>", "<ctrl>"}) == "<ctrl>+<shift>+x"

    def test_altgr_hides_implied_ctrl_alt(self):
        formatted = hotkeys.format_hotkey_tokens({"<alt_gr>", "<ctrl>", "<alt>", "q"})
        assert formatted == "<alt_gr>+q"


class TestPrettyHotkey:
    def test_empty_hotkey_renders_dash(self):
        assert hotkeys.pretty_hotkey("") == "—"


class TestVkMapping:
    def test_roundtrip_for_function_keys(self):
        for fkey in range(1, 13):
            token = f"<f{fkey}>"
            vk = hotkeys.token_to_windows_vk(token)
            assert vk is not None
            assert hotkeys.vk_to_key_token(vk) == token


    def test_unknown_vk_returns_none(self):
        assert hotkeys.vk_to_key_token(0xFF) is None
        assert hotkeys.vk_to_key_token(None) is None

    def test_altgr_implies_ctrl_alt(self):
        assert hotkeys.vk_to_hotkey_tokens(0xA5) == {"<alt_gr>", "<alt>", "<ctrl>"}

    def test_modifier_vks_collapse_to_generic_token(self):
        assert hotkeys.vk_to_hotkey_tokens(0xA2) == {"<ctrl>"}  # left ctrl
        assert hotkeys.vk_to_hotkey_tokens(0xA3) == {"<ctrl>"}  # right ctrl


class TestParseHotkeyBinding:
    def test_simple_modifier_plus_trigger(self):
        binding = hotkeys.parse_hotkey_binding("<ctrl>+<f9>", "transcription")
        assert binding is not None
        assert binding["action"] == "transcription"
        assert binding["modifiers"] == {"<ctrl>"}
        assert binding["trigger_tokens"] == {"<f9>"}
        assert binding["tokens"] == {"<ctrl>", "<f9>"}

    def test_empty_string_yields_none(self):
        assert hotkeys.parse_hotkey_binding("", "transcription") is None
        assert hotkeys.parse_hotkey_binding("+", "transcription") is None

    def test_windows_bindability(self):
        binding = hotkeys.parse_hotkey_binding("<ctrl>+<f9>", "t")
        assert binding is not None
        if is_WINDOWS:
            assert binding["windows_bindable"] is True
            assert binding["windows_vk"] == 0x78
        else:
            assert binding["windows_bindable"] is False

    def test_caps_lock_combo_is_never_os_bindable(self):
        binding = hotkeys.parse_hotkey_binding("<caps_lock>+<ctrl>", "t")
        assert binding is not None
        assert binding["windows_bindable"] is False


class TestBindingMatching:
    def _binding(self, hotkey: str) -> dict:
        binding = hotkeys.parse_hotkey_binding(hotkey, "test")
        assert binding is not None
        return binding

    def test_press_completes_binding(self):
        binding = self._binding("<ctrl>+<f9>")
        assert hotkeys.binding_matches_current_press(binding, {"<ctrl>", "<f9>"}, {"<f9>"})

    def test_press_without_modifier_does_not_match(self):
        binding = self._binding("<ctrl>+<f9>")
        assert not hotkeys.binding_matches_current_press(binding, {"<f9>"}, {"<f9>"})

    def test_pressed_set_matching(self):
        binding = self._binding("<ctrl>+<f9>")
        assert hotkeys.binding_matches_pressed(binding, {"<ctrl>", "<f9>"})
        assert not hotkeys.binding_matches_pressed(binding, {"<ctrl>"})

    def test_chord_needs_every_trigger_key(self):
        binding = self._binding("<f6>+<f7>")
        assert not hotkeys.binding_matches_current_press(binding, {"<f6>"}, {"<f6>"})
        assert hotkeys.binding_matches_current_press(binding, {"<f6>", "<f7>"}, {"<f7>"})
        assert not hotkeys.binding_matches_pressed(binding, {"<f7>"})
        chord = self._binding("<ctrl>+a+b")
        assert not hotkeys.binding_matches_current_press(chord, {"<ctrl>", "a"}, {"a"})
        assert hotkeys.binding_matches_current_press(chord, {"<ctrl>", "a", "b"}, {"b"})

    def test_modifier_only_combo_completes_on_its_last_modifier(self):
        binding = self._binding("<ctrl>+<alt>")
        assert not hotkeys.binding_matches_current_press(binding, {"<ctrl>"}, {"<ctrl>"})
        assert hotkeys.binding_matches_current_press(binding, {"<ctrl>", "<alt>"}, {"<alt>"})

    def test_release_tokens_expand_altgr(self):
        binding = self._binding("<alt_gr>+q")
        release = hotkeys.binding_release_tokens(binding)
        assert {"<alt_gr>", "<ctrl>", "<alt>", "q"}.issubset(release)

    def test_manual_suppression_for_caps_lock_and_char_triggers(self):
        assert hotkeys.binding_needs_manual_suppression(self._binding("<caps_lock>+<ctrl>"))
        assert hotkeys.binding_needs_manual_suppression(self._binding("<ctrl>+c"))
        assert not hotkeys.binding_needs_manual_suppression(self._binding("<ctrl>+<f9>"))


class TestClipboardShortcuts:
    def test_only_the_simulated_modifier_is_reserved(self):
        assert hotkeys.is_clipboard_shortcut("Ctrl+C", macos=False)
        assert hotkeys.is_clipboard_shortcut("<ctrl>+v", macos=False)
        assert hotkeys.is_clipboard_shortcut("<cmd>+a", macos=True)
        assert not hotkeys.is_clipboard_shortcut("<ctrl>+c", macos=True)  # macOS default stays valid
        assert not hotkeys.is_clipboard_shortcut("<ctrl>+<shift>+c", macos=False)
        assert not hotkeys.is_clipboard_shortcut("", macos=False)

