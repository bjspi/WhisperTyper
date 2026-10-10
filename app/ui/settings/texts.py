"""Declarative translations of the settings window: label, tooltip and placeholder tables."""
from __future__ import annotations

from html import escape
from typing import Any, Callable, NamedTuple, Optional, Tuple


class Text(NamedTuple):
    """A translated caption on ``attr`` and/or a shared tooltip on ``tooltip_attrs``."""

    key: Optional[str]
    attr: Optional[str] = None
    tooltip_key: Optional[str] = None
    tooltip_attrs: Tuple[str, ...] = ()
    template: str = "{}"
    # Rich tooltips repeat the caption as a bold title above the explanation.
    rich: bool = False


def _check(attr: str, key: str, tooltip_key: Optional[str] = None, rich: bool = False) -> Text:
    """A checkbox (or button) whose own tooltip explains it."""
    return Text(key, attr, tooltip_key, (attr,) if tooltip_key else (), rich=rich)


def _field(key: str, tooltip_key: Optional[str], label: str, *widgets: str) -> Text:
    """A label plus the controls it describes, all sharing one tooltip."""
    return Text(key, label, tooltip_key, (label, *widgets) if tooltip_key else ())


def _api_group_texts(provider_label: str, provider: str, model_label: str, model: str, url_label: str, url: str,
                     key_label: str, key: str, choose: str, add: str, temp_label: str, temp: str,
                     test: str, test_tooltip_key: str) -> Tuple[Text, ...]:
    """The same captions for both API groups; only the test button explains its own request."""
    return (
        Text("api_key_provider", provider_label, "api_provider_tooltip", (provider_label, provider), template="{}:"),
        _field("model_label", "model_tooltip", model_label, model),
        _field("api_url_label", "api_url_tooltip", url_label, url),
        _field("api_key_profile_label", "api_key_profile_tooltip", key_label, key),
        Text("api_key_choose_other", choose),
        Text("api_key_open_tab", add),
        _field("temperature_label", "temperature_tooltip", temp_label, temp),
        Text("test_connection_button", test, test_tooltip_key, (test,), template="🔌  {}"),
    )


TEXTS: Tuple[Text, ...] = (
    # Menus
    Text("menu_file", "file_menu"),
    Text("menu_file_open_config", "open_config_action"),
    Text("tray_log_action", "open_log_file_action"),
    Text("tray_play_action", "play_last_recording_action"),
    Text("menu_file_exit", "exit_action"),
    Text("menu_help", "help_menu"),
    Text("menu_help_about", "about_action"),
    Text("menu_help_github", "github_action"),
    # Transcription
    Text("transcription_api_group_title", "transcription_api_group"),
    *_api_group_texts("transcription_provider_label", "transcription_provider_selector", "model_label", "model_dropdown",
                      "api_endpoint_label", "api_endpoint_input", "api_key_label", "transcription_key_profile_selector",
                      "transcription_key_choose_button", "transcription_key_add_button",
                      "transcription_temp_label_title", "transcription_temp_slider",
                      "test_transcription_api_button", "test_connection_tooltip"),
    _field("ffmpeg_label", "ffmpeg_tooltip", "ffmpeg_label", "ffmpeg_path_input"),
    Text("ffmpeg_browse_button", "ffmpeg_browse_button"),
    _field("transcription_prompt_label", "transcription_prompt_tooltip", "transcription_prompt_label", "prompt_input"),
    # Recording
    Text("recording_group_title", "recording_group"),
    _field("hotkey_label", "hotkey_tooltip", "hotkey_label", "hotkey_display", "set_hotkey_button"),
    Text("set_hotkey_button", "set_hotkey_button"),
    _check("push_to_talk_checkbox", "push_to_talk_checkbox", "push_to_talk_tooltip"),
    _check("windows_keep_mic_hot_checkbox", "windows_keep_mic_hot_checkbox", "windows_keep_mic_hot_tooltip"),
    _field("windows_keep_mic_hot_idle_label", "windows_keep_mic_hot_idle_tooltip", "windows_keep_mic_hot_idle_label",
           "windows_keep_mic_hot_idle_input"),
    _field("min_recording_label", "min_recording_tooltip", "min_recording_label", "min_recording_input"),
    _field("recording_format_label", "recording_format_tooltip", "recording_format_label",
           "recording_format_selector", "recording_bitrate_label", "recording_bitrate_selector"),
    Text("recording_bitrate_label", "recording_bitrate_label"),
    _field("input_language_label", "input_language_tooltip", "input_language_label", "language_input"),
    _field("gain_label", "gain_tooltip", "gain_label", "gain_input"),
    # Rephrasing API
    Text("shared_api_group_title", "shared_api_group", "shared_api_group_tooltip", ("shared_api_group",)),
    *_api_group_texts("rephrasing_provider_label", "rephrasing_provider_selector", "rephrasing_model_label",
                      "rephrasing_model_input", "rephrasing_api_url_label", "rephrasing_api_url_input",
                      "rephrasing_api_key_label", "rephrasing_key_profile_selector",
                      "rephrasing_key_choose_button", "rephrasing_key_add_button",
                      "rephrasing_temp_label_title", "rephrasing_temp_slider",
                      "test_rephrasing_api_button", "test_api_button_tooltip"),
    # Prompts
    Text("transformations_tab_description", "transformations_tab_description_label"),
    Text("transformations_unavailable_message", "transformations_unavailable_label"),
    Text("caption_label", "caption_label"),
    _check("post_rp_show_during_recording_checkbox", "show_during_recording_checkbox", "show_during_recording_tooltip"),
    _check("post_rp_auto_apply_checkbox", "auto_apply_checkbox", "auto_apply_tooltip"),
    # Instruction (LivePrompt) entry; the text label follows the selected entry (editor).
    _check("liveprompt_enabled_checkbox", "liveprompt_enable_checkbox", "liveprompt_enable_tooltip"),
    Text(None, tooltip_key="liveprompt_help_button_tooltip", tooltip_attrs=("liveprompt_help_button",)),
    _field("liveprompt_trigger_label", "liveprompt_trigger_words_tooltip", "liveprompt_trigger_label",
           "liveprompt_trigger_words_input"),
    _field("liveprompt_trigger_scan_depth_label", "liveprompt_trigger_scan_depth_tooltip",
           "liveprompt_trigger_scan_depth_label", "liveprompt_trigger_scan_depth_input"),
    _check("liveprompt_strip_trigger_checkbox", "liveprompt_strip_trigger_checkbox", "liveprompt_strip_trigger_tooltip"),
    _check("rephrase_context_checkbox", "rephrase_context_checkbox", "rephrase_context_tooltip"),
    Text("add_button", "post_rp_add_btn"),
    Text("remove_button", "post_rp_remove_btn"),
    Text("post_rephrase_hotkey_group_title", "pr_hotkey_group"),
    Text("post_rephrase_hotkey_label", "pr_hotkey_label", "post_rephrase_hotkey_tooltip",
         ("pr_hotkey_group", "pr_hotkey_display", "set_pr_hotkey_button")),
    Text("set_hotkey_button", "set_pr_hotkey_button"),
    # General
    Text("ui_language_label", "ui_language_label"),
    Text("color_theme_label", "color_theme_label"),
    Text("input_device_label", "input_device_label"),
    _check("restore_clipboard_checkbox", "restore_clipboard_checkbox"),
    _check("debug_logging_checkbox", "debug_logging_checkbox"),
    _check("file_logging_checkbox", "file_logging_checkbox"),
    _check("redact_log_checkbox", "redact_log_checkbox", "redact_log_tooltip"),
    Text("proxy_url_label", "proxy_url_label", "proxy_url_tooltip", ("proxy_url_input",)),
    _check("use_px_proxy_checkbox", "use_px_proxy_checkbox", "use_px_proxy_tooltip"),
    Text("test_internet_button", "test_internet_button", "test_internet_tooltip",
         ("test_internet_button",), template="🌐  {}"),
    Text("log_retention_label", "log_retention_label", "log_retention_tooltip", ("log_retention_input",)),
    _check("systray_double_click_copy_checkbox", "systray_double_click_copy_checkbox"),
    _check("recording_prompt_overlay_system_position_checkbox", "recording_prompt_overlay_system_position_checkbox",
           "recording_prompt_overlay_system_position_tooltip"),
    _check("quit_without_confirmation_checkbox", "quit_without_confirmation_checkbox", "quit_without_confirmation_tooltip"),
    _check("play_g_button", "play_last_recording_button", "play_last_recording_tooltip"),
    _check("post_rephrase_auto_select_all_checkbox", "post_rephrase_auto_select_all_checkbox",
           "post_rephrase_auto_select_all_tooltip"),
    Text("misc_group_title", "misc_group"),
    Text("text_insertion_group_title", "text_insertion_group"),
    Text("logging_group_title", "logging_group"),
    _check("alt_clipboard_lib_checkbox", "alt_clipboard_lib_checkbox", "alt_clipboard_lib_tooltip", rich=True),
    _check("windows_sendinput_text_checkbox", "windows_sendinput_text_checkbox", "windows_sendinput_text_tooltip", rich=True),
    _check("windows_sendinput_fallback_checkbox", "windows_sendinput_fallback_checkbox",
           "windows_sendinput_fallback_tooltip", rich=True),
    _check("fast_paste_checkbox", "fast_paste_checkbox", "fast_paste_tooltip", rich=True),
    Text("save_button", "save_button"),
)

PLACEHOLDERS: Tuple[Tuple[str, str], ...] = (
    ("ffmpeg_path_input", "ffmpeg_path_placeholder"),
    ("prompt_input", "transcription_prompt_placeholder"),
    ("post_rp_text_edit", "text_placeholder"),
)


def _rich_tooltip(title: str, body: str) -> str:
    """Bold title above the explanation; paragraph breaks come from the translation itself."""
    return f"<qt><b>{escape(title)}</b><br><br>{escape(body).replace(chr(10), '<br>')}</qt>"


def apply_texts(window: Any, tr: Callable[..., str]) -> None:
    """Apply every caption, tooltip and placeholder of the tables to ``window``."""
    for text in TEXTS:
        caption = text.template.format(tr(text.key)) if text.key else ""
        if text.attr and text.key:
            widget = getattr(window, text.attr)
            (widget.setTitle if hasattr(widget, "setTitle") else widget.setText)(caption)
        if text.tooltip_key:
            tooltip = tr(text.tooltip_key)
            if text.rich:
                tooltip = _rich_tooltip(caption, tooltip)
            for attr in text.tooltip_attrs:
                getattr(window, attr).setToolTip(tooltip)
    for attr, key in PLACEHOLDERS:
        getattr(window, attr).setPlaceholderText(tr(key))
