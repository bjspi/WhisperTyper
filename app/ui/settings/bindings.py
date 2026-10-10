"""Declarative widget <-> config bindings: one table drives loading and saving the settings form."""
from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, MutableMapping, NamedTuple, Optional, Tuple

from app.core.env import is_MACOS, is_WINDOWS

#: Platform gates; a gated widget is hidden elsewhere and its config value is left untouched.
PLATFORMS: Dict[str, bool] = {
    "windows": is_WINDOWS,
    "windows_macos": is_WINDOWS or is_MACOS,
    "not_macos": not is_MACOS,
}


class Binding(NamedTuple):
    """One settings control and the config key it edits."""

    attr: str
    key: str
    kind: str
    platform: Optional[str] = None
    # Value forced on unsupported platforms (None: leave the config value as it is).
    unsupported_value: Any = None
    # Widgets shown/hidden with the control; for "temperature" the first one shows the value.
    companions: Tuple[str, ...] = ()


def _temperature_text(value: float) -> str:
    """Slider value label, e.g. "0.70"."""
    return f"{value:.2f}"


def _load_temperature(slider: Any, value: Any) -> None:
    """Temperatures 0.00-1.00 map onto a 0-100 slider."""
    slider.setRange(0, 100)
    slider.setValue(int(round(float(value) * 100)))


# kind -> (load(widget, value), save(widget) -> value)
_KINDS: Dict[str, Tuple[Callable[[Any, Any], None], Callable[[Any], Any]]] = {
    "check": (lambda w, v: w.setChecked(bool(v)), lambda w: w.isChecked()),
    "text": (lambda w, v: w.setText(str(v)), lambda w: w.text()),
    "stripped": (lambda w, v: w.setText(str(v)), lambda w: w.text().strip()),
    "plain": (lambda w, v: w.setPlainText(str(v)), lambda w: w.toPlainText()),
    "int": (lambda w, v: w.setValue(int(v)), lambda w: int(w.value())),
    "float": (lambda w, v: w.setValue(float(v)), lambda w: float(w.value())),
    "temperature": (_load_temperature, lambda w: w.value() / 100.0),
}

BINDINGS: Tuple[Binding, ...] = (
    # Transcription
    Binding("api_endpoint_input", "api_endpoint", "text"),
    Binding("transcription_temp_slider", "transcription_temperature", "temperature",
            companions=("transcription_temp_label",)),
    Binding("ffmpeg_path_input", "ffmpeg_path", "stripped"),
    Binding("prompt_input", "prompt", "plain"),
    # Recording
    Binding("push_to_talk_checkbox", "push_to_talk", "check"),
    Binding("windows_keep_mic_hot_checkbox", "windows_keep_mic_hot", "check", "windows"),
    Binding("windows_keep_mic_hot_idle_input", "windows_keep_mic_hot_idle_minutes", "int", "windows",
            companions=("windows_keep_mic_hot_idle_label",)),
    Binding("min_recording_input", "min_recording_seconds", "float"),
    # Rephrasing
    Binding("liveprompt_enabled_checkbox", "liveprompt_enabled", "check"),
    Binding("liveprompt_trigger_words_input", "liveprompt_trigger_words", "text"),
    Binding("liveprompt_trigger_scan_depth_input", "liveprompt_trigger_word_scan_depth", "int"),
    Binding("liveprompt_strip_trigger_checkbox", "liveprompt_strip_trigger", "check"),
    Binding("liveprompt_system_prompt_input", "liveprompt_system_prompt", "plain"),
    # Selected-text context needs permissions macOS does not grant reliably.
    Binding("rephrase_context_checkbox", "rephrase_use_selection_context", "check", "not_macos",
            unsupported_value=False),
    Binding("rephrasing_api_url_input", "rephrasing_api_url", "text"),
    Binding("rephrasing_temp_slider", "rephrasing_temperature", "temperature",
            companions=("rephrasing_temp_label",)),
    # General
    Binding("proxy_url_input", "proxy_url", "stripped"),
    Binding("use_px_proxy_checkbox", "use_local_px_proxy", "check"),
    Binding("log_retention_input", "log_retention_days", "int"),
    Binding("restore_clipboard_checkbox", "restore_clipboard", "check"),
    Binding("debug_logging_checkbox", "debug_logging", "check"),
    Binding("file_logging_checkbox", "file_logging", "check"),
    Binding("redact_log_checkbox", "redact_transcription_in_log", "check"),
    Binding("systray_double_click_copy_checkbox", "systray_double_click_copy", "check"),
    Binding("recording_prompt_overlay_system_position_checkbox", "recording_prompt_overlay_system_position", "check"),
    Binding("quit_without_confirmation_checkbox", "quit_without_confirmation", "check"),
    Binding("post_rephrase_auto_select_all_checkbox", "post_rephrase_auto_select_all", "check"),
    # Text insertion
    Binding("alt_clipboard_lib_checkbox", "alt_clipboard_lib", "check"),
    Binding("windows_sendinput_text_checkbox", "windows_sendinput_text", "check", "windows"),
    Binding("windows_sendinput_fallback_checkbox", "windows_sendinput_fallback", "check", "windows"),
    Binding("fast_paste_checkbox", "fast_paste", "check", "windows_macos"),
)


def _supported(binding: Binding) -> bool:
    """Whether the binding's control exists on this platform."""
    return binding.platform is None or PLATFORMS[binding.platform]


def load_bindings(window: Any, config: Mapping[str, Any], bindings: Tuple[Binding, ...] = BINDINGS) -> None:
    """Fill every bound control from ``config`` and hide controls of other platforms."""
    for binding in bindings:
        widget = getattr(window, binding.attr)
        supported = _supported(binding)
        value = config[binding.key] if supported or binding.unsupported_value is None else binding.unsupported_value
        _KINDS[binding.kind][0](widget, value)
        companions = [getattr(window, attr) for attr in binding.companions]
        if not supported:
            for control in (widget, *companions):
                control.setVisible(False)
        if binding.kind == "temperature":
            label = companions[0]
            label.setText(_temperature_text(float(value)))
            widget.valueChanged.connect(lambda raw, label=label: label.setText(_temperature_text(raw / 100.0)))


def save_bindings(window: Any, config: MutableMapping[str, Any], bindings: Tuple[Binding, ...] = BINDINGS) -> None:
    """Write every bound control back to ``config``; other platforms' values stay as saved."""
    for binding in bindings:
        if _supported(binding):
            config[binding.key] = _KINDS[binding.kind][1](getattr(window, binding.attr))
        elif binding.unsupported_value is not None:
            config[binding.key] = binding.unsupported_value
