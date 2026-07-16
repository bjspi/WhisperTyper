"""Application constants, config defaults, and system-language detection.

Single responsibility: hold static configuration data and the small pure helpers that
compute launch-time defaults. No Qt, no app state.
"""

from __future__ import annotations

import glob
import logging
import os
import subprocess
from typing import Any, Dict

from app.core.env import is_MACOS, is_WINDOWS
from app.core.prompts import (
    DEFAULT_GENERIC_REPHRASE_PROMPTS,
    DEFAULT_LIVEPROMPT_SYSTEM_PROMPTS,
    DEFAULT_TRANSCRIPTION_PROMPTS,
    _default_prompt_for,
)


def get_system_language_2char() -> str:
    """Return a 2-character language code (e.g. 'en', 'de'); English variants map to 'en'."""
    lang = None
    if is_WINDOWS:
        try:
            import ctypes
            windll = ctypes.windll.kernel32
            lang_id = windll.GetUserDefaultUILanguage()
            import locale
            lang = locale.windows_locale.get(lang_id, 'en')
        except Exception:
            lang = os.environ.get('LANG', 'en')
    elif is_MACOS:
        try:
            output = subprocess.check_output(
                ["defaults", "read", "-g", "AppleLanguages"],
                universal_newlines=True
            )
            import re
            match = re.search(r'"([a-zA-Z\-]+)"', output)
            if match:
                lang = match.group(1)
        except Exception:
            lang = os.environ.get('LANG', 'en')
    else:
        lang = os.environ.get('LANG', 'en')

    if not lang:
        return 'en'
    lang = lang.replace('_', '-').lower()
    if lang.startswith('en'):
        return 'en'
    return lang.split('-')[0]


# On macOS, showing a floating window steals focus; these re-activate the previous app.
def get_active_app_name() -> str:
    """Return the name of the frontmost macOS application."""
    script = 'tell application "System Events" to get name of first application process whose frontmost is true'
    return subprocess.check_output(['osascript', '-e', script]).decode().strip()


def activate_app(app_name: str) -> None:
    """Bring the named macOS application back to the foreground."""
    script = f'tell application "{app_name}" to activate'
    subprocess.call(['osascript', '-e', script])


SYS_LANG = get_system_language_2char()
logging.info(f"Detected system language: {SYS_LANG}")

TRANSCRIPTION_MODEL_OPTIONS = [
    "whisper-1 (openai)",
    "gpt-4o-transcribe (openai)",
    "gpt-4o-mini-transcribe (openai)",
    "whisper-large-v3 (groq)",
    "whisper-large-v3-turbo (groq)",
    "Custom"
]
DEFAULT_TRANSCRIPTION_MODEL = "whisper-1 (openai)"
DEFAULT_REPHRASING_MODEL = "gpt-5.6-luna"
# Existing configs still on one of these built-in defaults are migrated to the current default.
PREVIOUS_DEFAULT_REPHRASING_MODELS = {"gpt-4o-mini", "gpt-5.4", "gpt-5.5", "gpt-5.6"}
# Approximate max token length for the Whisper initial prompt.
WHISPER_PROMPT_TOKEN_LIMIT = 230

# Map display names to ISO 639-1 codes
LANGUAGES = {
    "Detect Language": "",
    "English": "en", "German": "de", "French": "fr", "Spanish": "es", "Italian": "it", "Dutch": "nl",
    "Afrikaans": "af", "Arabic": "ar", "Armenian": "hy", "Azerbaijani": "az", "Belarusian": "be",
    "Bosnian": "bs", "Bulgarian": "bg", "Catalan": "ca", "Chinese": "zh", "Croatian": "hr",
    "Czech": "cs", "Danish": "da", "Estonian": "et", "Finnish": "fi", "Galician": "gl",
    "Greek": "el", "Hebrew": "he", "Hindi": "hi", "Hungarian": "hu", "Icelandic": "is",
    "Indonesian": "id", "Japanese": "ja", "Kannada": "kn", "Kazakh": "kk", "Korean": "ko",
    "Latvian": "lv", "Lithuanian": "lt", "Macedonian": "mk", "Malay": "ms", "Marathi": "mr",
    "Maori": "mi", "Nepali": "ne", "Norwegian": "no", "Persian": "fa", "Polish": "pl",
    "Portuguese": "pt", "Romanian": "ro", "Russian": "ru", "Serbian": "sr", "Slovak": "sk",
    "Slovenian": "sl", "Swahili": "sw", "Swedish": "sv", "Tagalog": "tl", "Tamil": "ta",
    "Thai": "th", "Turkish": "tr", "Ukrainian": "uk", "Urdu": "ur", "Vietnamese": "vi", "Welsh": "cy"
}

USER_HOME_DIR = os.path.expanduser("~")
APP_DATA_DIR = os.path.join(USER_HOME_DIR, ".WhisperTyper")
os.makedirs(APP_DATA_DIR, exist_ok=True)

CONFIG_FILE: str = os.path.join(APP_DATA_DIR, "config.json")
LOG_FILE_PATH: str = os.path.join(APP_DATA_DIR, "WhisperTyper.log")

# lang JSON files live in app/lang (this module is app/core/constants.py -> ../lang)
UI_LANG_FILES = [
    os.path.splitext(os.path.basename(x))[0]
    for x in glob.glob(os.path.join(os.path.dirname(__file__), '..', 'lang', '*.json'))
]

# Default UI language used to pick the initial default prompts.
_DEFAULT_UI_LANG = SYS_LANG if SYS_LANG in UI_LANG_FILES else "en"

CONFIG_SCHEMA_VERSION = 1

# Window sizing: default on fresh install + the enforced minimum the user can't shrink past.
# The minimum is sized so the tallest settings page (Rephrasing) shows fully without scrolling.
WINDOW_MIN_WIDTH = 680
WINDOW_MIN_HEIGHT = 1080

DEFAULT_CONFIG: Dict[str, Any] = {
    "config_schema_version": CONFIG_SCHEMA_VERSION,
    "color_theme": "system",  # "system" (auto light/dark) | "light" | "dark"
    "input_device_name": "",  # "" = system default input; else a device name from the dropdown
    "window_width": 760,
    "window_height": 1080,
    "api_key": "",
    "api_endpoint": "https://api.openai.com/v1/audio/transcriptions",
    "model": DEFAULT_TRANSCRIPTION_MODEL,
    "transcription_temperature": 0.0,
    # Path to an ffmpeg binary (or the folder containing it). Empty = auto-detect on PATH.
    # When ffmpeg is available, video files can be picked for transcription (audio is extracted).
    "ffmpeg_path": "",
    # Last folder used in the "Transcribe audio file(s)…" picker, so it reopens there next time.
    "last_transcribe_dir": "",
    # Upload-size ceiling (MB). Files above this are compressed with ffmpeg (mono/16 kHz, bitrate
    # lowered as needed) to fit. Default is safely under Groq's 25 MB free-tier limit; raise it for
    # OpenAI (25 MB) / Groq dev tier (100 MB). Needs ffmpeg to actually compress oversized files.
    "max_upload_mb": 24,
    # Floor for that compression. The bitrate is only dropped this low when a file would otherwise
    # exceed max_upload_mb; if even this won't fit, the file is rejected with an error.
    "min_audio_bitrate_kbps": 80,
    "prompt": _default_prompt_for(DEFAULT_TRANSCRIPTION_PROMPTS, _DEFAULT_UI_LANG),
    # Stored in canonical token form (see app/core/hotkeys.py) so a fresh config
    # loads without an immediate normalization rewrite.
    "hotkey": "<ctrl>+x" if is_MACOS else "<caps_lock>+<ctrl>",
    "push_to_talk": False,
    "windows_keep_mic_hot": is_WINDOWS,
    "windows_keep_mic_hot_idle_minutes": 15,
    # Recordings shorter than this (seconds) are treated as a mis-tap and discarded without
    # transcribing. 0 disables the guard. Allows an instant "oops" cancel by tapping again.
    "min_recording_seconds": 1.0,
    "input_language": SYS_LANG if SYS_LANG in LANGUAGES.values() else "en",
    "ui_language": _DEFAULT_UI_LANG,
    "restore_clipboard": True,
    "debug_logging": True,
    "file_logging": True,
    "redact_transcription_in_log": True,
    "log_retention_days": 3,
    "proxy_url": "",
    "use_local_px_proxy": False,
    "gain_db": 10,
    "systray_double_click_copy": True,
    "quit_without_confirmation": False,
    "alt_clipboard_lib": is_MACOS,

    # Rephrasing / LivePrompt
    "liveprompt_enabled": True,
    "liveprompt_trigger_words": "prompt, ",
    "liveprompt_trigger_word_scan_depth": 5,
    # When True, strip everything up to and including the trigger word before rephrasing, so
    # only the actual instruction after it is sent (e.g. "… als Anweisung. Schreib …" -> "Schreib …").
    "liveprompt_strip_trigger": False,
    "liveprompt_system_prompt": _default_prompt_for(DEFAULT_LIVEPROMPT_SYSTEM_PROMPTS, _DEFAULT_UI_LANG),
    "rephrase_use_selection_context": False,
    "generic_rephrase_enabled": False,
    "generic_rephrase_prompt": _default_prompt_for(DEFAULT_GENERIC_REPHRASE_PROMPTS, _DEFAULT_UI_LANG),
    "rephrasing_api_url": "https://api.openai.com/v1/chat/completions",
    "rephrasing_api_key": "",
    "rephrasing_model": DEFAULT_REPHRASING_MODEL,
    "rephrasing_temperature": 0.7,
    "post_rephrasing_entries": [],
    "post_rephrase_hotkey": "<ctrl>+c" if is_MACOS else "<f9>",
    "post_rephrase_auto_select_all": False,
    # macOS Permissions
    "macos_input_monitoring_info_shown": False,
    "macos_accessibility_info_shown": False,
    "macos_microphone_info_shown": False,
}
