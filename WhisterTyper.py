import subprocess
import sys
import os
import re
import json
import threading
import tempfile
import time
from datetime import datetime
from typing import List, Dict, Any, Set, Optional

# GUI and System Tray
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit,
                             QPushButton, QSystemTrayIcon, QMenu,
                             QMessageBox, QTextEdit, QStyle, QHBoxLayout, QComboBox, QCheckBox, QTabWidget, QScrollArea,
                             QFrame, QListWidget, QListWidgetItem, QSplitter, QAbstractItemView, QSlider)
from PyQt6.QtCore import QObject, pyqtSignal, QThread, QTimer, Qt, QPoint
from PyQt6.QtGui import QIcon, QCloseEvent, QCursor, QAction

# Global Hotkey
from pynput import keyboard

# Audio Recording & Playback with PyAudio (replaces sounddevice/numpy/scipy/pygame)
import pyaudio
import wave
import struct
import math

# API Request and Text Output
import requests
import pyautogui
import copykitten

# Logging
import logging

class TranslationManager:
    """Manages loading and retrieving translated strings from JSON files."""

    def __init__(self, initial_language: str = 'en'):
        """
        Initializes the TranslationManager.

        Args:
            initial_language (str): The initial language code (e.g., 'en').
        """
        self.translations: Dict[str, str] = {}
        self.language: str = ''
        self.set_language(initial_language)

    def get_base_path(self) -> str:
        """
        Gets the base path for resource files, compatible with PyInstaller.

        Returns:
            str: The absolute base path.
        """
        if getattr(sys, 'frozen', False):
            return sys._MEIPASS
        else:
            # In a normal environment, this file is in the project root.
            return os.path.dirname(__file__)

    def set_language(self, lang_code: str) -> None:
        """
        Sets the current language and loads the corresponding translation file.

        Args:
            lang_code (str): The language code to load (e.g., 'en', 'de').
        """
        if self.language == lang_code:
            return

        base_path = self.get_base_path()
        translations_dir = os.path.join(base_path, 'lang')
        filepath = os.path.join(translations_dir, f"{lang_code}.json")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.translations = json.load(f)
            self.language = lang_code
            print(f"Successfully loaded language: {lang_code}")
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Could not load language file for '{lang_code}'. Falling back to English.")
            # Fallback to English if the selected language file is missing/corrupt
            if lang_code != 'en':
                self.set_language('en')

    def tr(self, key: str, **kwargs: Any) -> str:
        """
        Retrieves a translated string for a given key.

        Args:
            key (str): The key for the string to translate.
            **kwargs: Placeholder values to format into the string.

        Returns:
            str: The translated and formatted string, or the key if not found.
        """
        text = self.translations.get(key, key)
        if kwargs:
            try:
                return text.format(**kwargs)
            except KeyError:
                return text  # Return raw text if format keys don't match
        return text

# --- Default Prompts ---
DEFAULT_TRANSCRIPTION_PROMPT = """
The following is a transcription of a voice input. The transcription should be almost perfect to the original, only filler words and silence/emptiness should be removed. Please pay attention to spelling, capitalization, and sensible punctuation, including periods and commas. I also use "Germanized" English terms, especially from the tech and IT scene, from the areas of gadgets, smartphones, automotive, AI, and Python programming. Please recognize these as well.
"""

DEFAULT_REPHRASING_PROMPT = """
I will give you a transcript from a user. You should either return the text in its original form with revised formatting OR follow the prompt. If the following text explicitly contains the word 'prompt' at thebeginning, consider the text as a prompt, follow it and its instructions, and write a text from it. If you do not find any instructions to use it as a prompt, then simply return the TEXT to me in its original form, replacing ONLY "new line" with a line break."""

TRANSCRIPTION_MODEL_OPTIONS = [
    "whisper-1 (openai)",
    "gpt-4o-transcribe (openai)",
    "gpt-4o-mini-transcribe (openai)",
    "whisper-large-v3 (groq)",
    "whisper-large-v3-turbo (groq)",
    "Custom"
]
DEFAULT_TRANSCRIPTION_MODEL = "whisper-1 (openai)"
DEFAULT_REPHRASING_MODEL = "gpt-4o-mini"
# Approximate max token length for Whisper initial prompt (variously documented ~224; using 230 for safety margin display)
WHISPER_PROMPT_TOKEN_LIMIT = 230

# --- Languages ---
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

# --- Configuration ---
# Define the application's base directory for both frozen and non-frozen states
APP_BASE_DIR = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(__file__)
CONFIG_FILE: str = os.path.join(APP_BASE_DIR, "config.json")

DEFAULT_HOTKEY_STR: str = "<f9>"

DEFAULT_CONFIG: Dict[str, Any] = {
    "api_key": "",
    "api_endpoint": "https://api.openai.com/v1/audio/transcriptions",
    "model": DEFAULT_TRANSCRIPTION_MODEL,
    "transcription_temperature": 0.0,
    "prompt": DEFAULT_TRANSCRIPTION_PROMPT,
    "hotkey": DEFAULT_HOTKEY_STR,
    "input_language": "en",
    "ui_language": "en",
    "restore_clipboard": True,
    "debug_logging": True,
    "file_logging": False,
    "gain_db": 0,
    "rephrasing_enabled": True,
    "rephrase_use_selection_context": True,
    "rephrasing_prompt": DEFAULT_REPHRASING_PROMPT,
    "rephrasing_trigger_word": "prompt, liveprompt",
    "rephrasing_api_url": "https://api.openai.com/v1/chat/completions",
    "rephrasing_api_key": "",
    "rephrasing_model": DEFAULT_REPHRASING_MODEL,
    "rephrasing_temperature": 0.7,
    "post_rephrasing_entries": []
}

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
is_MACOS = sys.platform.startswith('darwin')
is_WINDOWS = sys.platform.startswith('win')

class MouseFollowerTooltip(QWidget):
    """
    A custom frameless widget that follows the mouse cursor and closes after a timeout.
    This class is designed to be instantiated and managed via its static 'show_tooltip' method.
    """
    _instance: Optional['MouseFollowerTooltip'] = None
    _close_timer: Optional[QTimer] = None
    _move_timer: Optional[QTimer] = None

    def __init__(self, message: str, timeout_ms: int = 2000):
        """
        Initializes the tooltip widget.

        Args:
            message (str): The text message to display in the tooltip.
            timeout_ms (int): The duration in milliseconds before the tooltip automatically closes.
        """
        super().__init__()
        # Set window flags to create a frameless, top-level tooltip
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.ToolTip | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        self.label = QLabel(message, self)
        self.label.setStyleSheet("""
            background-color: rgba(255, 255, 224, 0.95); /* Semi-transparent yellow */
            color: black;
            border: 1px solid black;
            padding: 5px;
            border-radius: 3px;
        """)
        self.label.adjustSize()
        self.resize(self.label.size())

        self.move_to_mouse()
        self.show()

        # Timer to close the tooltip after the specified timeout
        if MouseFollowerTooltip._close_timer:
            MouseFollowerTooltip._close_timer.stop()
        MouseFollowerTooltip._close_timer = QTimer(self)
        MouseFollowerTooltip._close_timer.setSingleShot(True)
        MouseFollowerTooltip._close_timer.timeout.connect(self.close)  # Directly connect to close
        MouseFollowerTooltip._close_timer.start(timeout_ms)

        # Timer to update the tooltip's position to follow the mouse
        if MouseFollowerTooltip._move_timer:
            MouseFollowerTooltip._move_timer.stop()
        MouseFollowerTooltip._move_timer = QTimer(self)
        MouseFollowerTooltip._move_timer.timeout.connect(self.move_to_mouse)
        MouseFollowerTooltip._move_timer.start(16)  # ~60 FPS update rate

        MouseFollowerTooltip._instance = self

    def move_to_mouse(self) -> None:
        """Moves the tooltip to the current cursor position with a slight offset."""
        pos = QCursor.pos() + QPoint(15, 15)
        self.move(pos)

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Overrides the close event to ensure timers are stopped and the static instance is cleared.

        Args:
            event (QCloseEvent): The close event.
        """
        # Stop timers to prevent them from running after the widget is gone
        if MouseFollowerTooltip._move_timer:
            MouseFollowerTooltip._move_timer.stop()
            MouseFollowerTooltip._move_timer = None
        if MouseFollowerTooltip._close_timer:
            MouseFollowerTooltip._close_timer.stop()
            MouseFollowerTooltip._close_timer = None

        # Clear the static instance reference if it's this instance
        if MouseFollowerTooltip._instance is self:
            MouseFollowerTooltip._instance = None

        event.accept()

    @staticmethod
    def show_tooltip(message: str, timeout_ms: int = 2000) -> None:
        """
        Displays a new tooltip. If one is already visible, it is closed first.

        Args:
            message (str): The message to display.
            timeout_ms (int): How long the tooltip should be visible in milliseconds.
        """
        # Close the previous tooltip if it exists
        if MouseFollowerTooltip._instance:
            MouseFollowerTooltip._instance.close()

        # Create a new instance
        MouseFollowerTooltip(message, timeout_ms)

# ---------------- Helper Token Utilities ----------------
TOKEN_PATTERN = re.compile(r"\w+|[^\s\w]")  # crude approximation (words or single punctuation)

def estimate_tokens(text: str) -> int:
    """Very lightweight approximate tokenizer.
    NOTE: This is NOT identical to Whisper's exact tokenizer but good enough for length validation.
    """
    if not text:
        return 0
    return len(TOKEN_PATTERN.findall(text))

# Function which searches a Ressource file either in the PyInstalled Temp folder or in the current directory of the script.
def resource_path(relative_path: str) -> str:
    """
    Get the absolute path to a resource, works for both frozen and non-frozen applications.

    Args:
        relative_path (str): The relative path to the resource.

    Returns:
        str: The absolute path to the resource.
    """
    if getattr(sys, 'frozen', False):
        # If the application is frozen (e.g., using PyInstaller), resources are in a temp folder.
        base_path = sys._MEIPASS
    else:
        # If the application is not frozen, resources are relative to the script.
        base_path = os.path.dirname(__file__)
    return os.path.join(base_path, relative_path)

class TranscriptionWorker(QObject):
    """
    Runs the API request in a separate thread to avoid blocking the GUI.
    """
    finished = pyqtSignal(str)
    error = pyqtSignal(str, str)

    def __init__(self, api_key: str, api_endpoint: str, audio_path: str, prompt: str, model: str,
                 language: str, temperature: float) -> None:
        """
        Initializes the transcription worker.

        Args:
            api_key (str): The API key for the transcription service.
            api_endpoint (str): The URL of the transcription API endpoint.
            audio_path (str): The local path to the audio file to be transcribed.
            prompt (str): A prompt to guide the transcription model.
            model (str): The name of the transcription model to use.
            language (str): The language of the audio in ISO 639-1 format (e.g., "en", "de"). Can be empty for auto-detection.
            temperature (float): The sampling temperature for the model.
        """
        super().__init__()
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.audio_path = audio_path
        self.prompt = prompt
        self.model = re.sub(r"\s*\(.*?\)", "", model).strip()
        self.language = language.lower() if language else ""
        self.temperature = temperature

    def run(self) -> None:
        """
        Executes the transcription request and emits the corresponding signal.
        """
        logging.info("TranscriptionWorker started.")
        try:
            if not self.api_key:
                logging.debug("No API key provided in configuration.")
                raise ValueError("API key not found in configuration.")

            headers: Dict[str, str] = {"Authorization": f"Bearer {self.api_key}"}
            data: Dict[str, Any] = {"model": self.model, "prompt": self.prompt, "temperature": self.temperature}
            # Only add language if it's not empty (for auto-detection)
            if self.language:
                data["language"] = self.language

            logging.debug(f"API endpoint: {self.api_endpoint}")
            logging.debug(f"Request data: {data}")
            logging.debug(f"Audio file path: {self.audio_path}")

            with open(self.audio_path, 'rb') as audio_file:
                files = {"file": (os.path.basename(self.audio_path), audio_file, "audio/wav")}
                logging.debug(f"Sending POST request to API with files: {files['file'][0]}")
                response = requests.post(self.api_endpoint, headers=headers, files=files, data=data)

            logging.debug(f"API response status: {response.status_code}")
            if response.status_code == 200:
                transcription: str = response.json().get("text", "")
                logging.info(f"Transcription result: {transcription}")
                self.finished.emit(transcription)
            else:
                error_msg = f"API Error: {response.status_code}\n{response.text}"
                logging.error(error_msg)
                self.error.emit(error_msg, self.audio_path)
        except Exception as e:
            error_msg = f"An unexpected error occurred in worker:\n{str(e)}"
            logging.error(error_msg)
            self.error.emit(error_msg, self.audio_path)

class VoiceTranscriberApp(QWidget):
    """
    Main application class for the Voice Transcriber.
    Manages the GUI, system tray icon, audio recording, and hotkey listener.
    """
    # CORRECTED: Signal for thread-safe GUI updates
    show_tooltip_signal = pyqtSignal(str, int)

    def __init__(self) -> None:
        """Initializes the application."""
        super().__init__()
        self.is_recording: bool = False
        self.recorded_frames: List[bytes] = []
        self.samplerate: int = 16000
        self.chunk_size: int = 1024
        # self.pygame_initialized: bool = False  # removed
        self.pyaudio_instance: Optional[pyaudio.PyAudio] = None

        self.config: Dict[str, Any] = {}
        self.load_config()  # Load config first to get UI language

        # Initialize TranslationManager
        self.translator = TranslationManager(initial_language=self.config.get('ui_language', 'en'))

        # Add placeholder for file log handler
        self._file_log_handler = None

        self.hotkey_str: str = DEFAULT_HOTKEY_STR
        self.new_hotkey_str: Optional[str] = None
        self.captured_keys: Set[Any] = set()

        self.recording_thread: Optional[threading.Thread] = None
        self.hotkey_listener: Optional[keyboard.GlobalHotKeys] = None
        self.hotkey_capture_listener: Optional[keyboard.Listener] = None
        self.active_workers: List[TranscriptionWorker] = []
        self.active_threads: List[QThread] = []
        self.last_transcription: str = ""
        self.current_transcription_context: str = ""

        self.cleanup_old_recordings()
        # self.load_config() # Moved up before translator init
        # After loading config ensure logging handlers reflect settings
        self.apply_logging_configuration()
        self.init_ui()
        self.init_tray_icon()
        # self.init_audio_player()  # removed pygame init
        self.init_manual_hotkey_listener()
        # Preload short sound effects & prepare reusable output streams for low latency
        self.sound_cache: Dict[str, tuple] = {}
        self._output_streams: Dict[tuple, Any] = {}
        self.preload_sounds()
        self._preopen_streams()  # Pre-open streams for cached sounds

        # Connect the signal to the slot for safe cross-thread communication
        self.show_tooltip_signal.connect(self._show_tooltip_slot)

        self.tray_icon.setToolTip(self.translator.tr("tray_ready_tooltip", hotkey=self.hotkey_str))
        logging.info(f"Application started. Press '{self.hotkey_str}' to start/stop recording.")
        self.show_tray_balloon(self.translator.tr("tray_started_message"), 2000)

    def _show_tooltip_slot(self, message: str, timeout_ms: int) -> None:
        """
        This slot is executed in the main GUI thread and can safely update the UI.

        Args:
            message (str): The message to display in the tooltip.
            timeout_ms (int): The duration in milliseconds for the tooltip to be visible.
        """
        MouseFollowerTooltip.show_tooltip(message, timeout_ms)

    def show_tray_balloon(self, message: str, timeout_ms: int = 2000) -> None:
        """
        Shows a custom tooltip by emitting a signal to the main thread.
        This is the thread-safe way to show tooltips from any thread.

        Args:
            message (str): The message to display.
            timeout_ms (int): The duration in milliseconds.
        """
        self.show_tooltip_signal.emit(message, timeout_ms)

    def load_config(self) -> None:
        """
        Loads configuration from JSON file.
        Ensures all default keys are present, and saves back if any were added.
        """
        loaded_config = {}
        try:
            with open(CONFIG_FILE, 'r') as f:
                loaded_config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # File doesn't exist or is corrupted, will proceed with defaults
            pass

        config_updated = False

        # --- Start Migration ---
        # Migrate old 'language' key to 'input_language'
        if "language" in loaded_config and "input_language" not in loaded_config:
            loaded_config["input_language"] = loaded_config.pop("language")
            config_updated = True

        # Check if input_language is a display name and convert to code
        if "input_language" in loaded_config:
            lang_value = loaded_config["input_language"]
            # If it's a name (e.g., "German", length > 2), convert it to code
            if len(lang_value) > 2 and lang_value in LANGUAGES:
                loaded_config["input_language"] = LANGUAGES[lang_value]
                config_updated = True
        # --- End Migration ---

        # Ensure all default keys exist in the loaded config
        for key, default_value in DEFAULT_CONFIG.items():
            if key not in loaded_config:
                loaded_config[key] = default_value
                config_updated = True

        self.config = loaded_config
        self.hotkey_str = self.config["hotkey"]

        # If we added any missing keys, save the file back
        if config_updated:
            self.save_config()

    def save_config(self) -> None:
        """Saves the current configuration to the JSON file."""
        # Ensure the ressources directory exists
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f, indent=4)
        logging.info("Configuration saved.")

    def init_ui(self) -> None:
        """Initializes the settings window with tabs."""
        self.setWindowTitle("Voice Transcriber - Settings")

        # Main layout for the entire window. By passing 'self' to the constructor,
        # this layout is automatically set for the VoiceTranscriberApp widget.
        main_layout = QVBoxLayout(self)

        # Create the tab widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Create widgets for each tab
        transcription_tab = QWidget()
        rephrasing_tab = QWidget()
        post_rephrasing_tab = QWidget()
        general_tab = QWidget()

        # Add tabs to the tab widget
        self.tabs.addTab(transcription_tab, "") # Text set in retranslate_ui
        self.tabs.addTab(rephrasing_tab, "")
        self.tabs.addTab(post_rephrasing_tab, "")
        self.tabs.addTab(general_tab, "")

        # --- Populate Transcription Tab ---
        transcription_layout = QVBoxLayout(transcription_tab)

        self.api_key_label = QLabel()
        transcription_layout.addWidget(self.api_key_label)
        self.api_key_input = QLineEdit(self.config["api_key"])
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        transcription_layout.addWidget(self.api_key_input)

        self.api_endpoint_label = QLabel()
        transcription_layout.addWidget(self.api_endpoint_label)
        self.api_endpoint_input = QLineEdit(self.config["api_endpoint"])
        transcription_layout.addWidget(self.api_endpoint_input)

        api_button_layout = QHBoxLayout()
        self.openai_button = QPushButton()
        self.groq_button = QPushButton()
        api_button_layout.addWidget(self.openai_button)
        api_button_layout.addWidget(self.groq_button)
        api_button_layout.addStretch(1)
        transcription_layout.addLayout(api_button_layout)
        self.openai_button.clicked.connect(
            lambda: self.api_endpoint_input.setText("https://api.openai.com/v1/audio/transcriptions"))
        self.groq_button.clicked.connect(
            lambda: self.api_endpoint_input.setText("https://api.groq.com/openai/v1/audio/transcriptions"))

        self.model_label = QLabel()
        transcription_layout.addWidget(self.model_label)
        model_layout = QHBoxLayout()
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(TRANSCRIPTION_MODEL_OPTIONS)
        model_layout.addWidget(self.model_dropdown)
        self.model_input = QLineEdit(visible=False)
        model_layout.addWidget(self.model_input)
        transcription_layout.addLayout(model_layout)
        model_value = self.config["model"]
        if self.model_dropdown.findText(model_value) != -1:
            self.model_dropdown.setCurrentText(model_value)
        else:
            self.model_dropdown.setCurrentText("Custom")
            self.model_input.setText(model_value)
            self.model_input.setVisible(True)
        self.model_dropdown.currentTextChanged.connect(lambda text: self.model_input.setVisible(text == "Custom"))
        # Update token counter when model changes (could affect limit applicability)
        self.model_dropdown.currentTextChanged.connect(lambda _t: self._update_prompt_token_counter())
        self.model_input.textChanged.connect(lambda _t: self._update_prompt_token_counter())

        # --- Temperature Slider ---
        self.transcription_temp_label_title = QLabel()
        transcription_layout.addWidget(self.transcription_temp_label_title)
        temp_layout = QHBoxLayout()
        self.transcription_temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.transcription_temp_slider.setRange(0, 100)
        self.transcription_temp_slider.setValue(int(self.config["transcription_temperature"] * 100))
        self.transcription_temp_label = QLabel(f"{self.config['transcription_temperature']:.2f}")
        self.transcription_temp_slider.valueChanged.connect(self._update_transcription_temp_label)
        temp_layout.addWidget(self.transcription_temp_slider)
        temp_layout.addWidget(self.transcription_temp_label)
        transcription_layout.addLayout(temp_layout)

        # --- Hotkey, Language, and Gain Side-by-Side ---
        controls_layout = QHBoxLayout()

        # Left side: Hotkey
        hotkey_v_layout = QVBoxLayout()
        self.hotkey_label = QLabel()
        hotkey_v_layout.addWidget(self.hotkey_label)
        hotkey_h_layout = QHBoxLayout()
        self.hotkey_display = QLineEdit(self.hotkey_str)
        self.hotkey_display.setReadOnly(True)
        self.set_hotkey_button = QPushButton()
        self.set_hotkey_button.clicked.connect(self.start_hotkey_capture)
        hotkey_h_layout.addWidget(self.hotkey_display)
        hotkey_h_layout.addWidget(self.set_hotkey_button)
        hotkey_v_layout.addLayout(hotkey_h_layout)
        controls_layout.addLayout(hotkey_v_layout)

        # Middle: Language
        lang_v_layout = QVBoxLayout()
        self.input_language_label = QLabel()
        lang_v_layout.addWidget(self.input_language_label)
        self.language_input = QComboBox()
        # Create a reverse map from ISO code to display name for easy lookup
        self.lang_code_to_name = {v: k for k, v in LANGUAGES.items()}
        self.language_input.addItems(LANGUAGES.keys())
        # Set current value based on config
        lang_code = self.config["input_language"].lower()
        display_name = self.lang_code_to_name.get(lang_code, "English") # Default to English if not found
        self.language_input.setCurrentText(display_name)
        lang_v_layout.addWidget(self.language_input)
        controls_layout.addLayout(lang_v_layout)

        # Right side: Gain
        gain_v_layout = QVBoxLayout()
        self.gain_label = QLabel()
        gain_v_layout.addWidget(self.gain_label)
        # Ensure gain is a float and convert to string for display
        self.gain_input = QLineEdit(str(self.config["gain_db"]))
        self.gain_input.setPlaceholderText("0")
        # Set Tooltip on the Gain Widget:
        self.gain_input.setToolTip(
            "Adjust the gain for the recorded WAV file. "
            "Positive values amplify the sound, negative values reduce it. "
            "0 dB means no change. You can listen to the recorded audio to find the best setting depending on your hardware!"
        )
        gain_v_layout.addWidget(self.gain_input)
        controls_layout.addLayout(gain_v_layout)

        transcription_layout.addLayout(controls_layout)

        self.transcription_prompt_label = QLabel()
        transcription_layout.addWidget(self.transcription_prompt_label)
        self.prompt_input = QTextEdit(self.config["prompt"], placeholderText="Enter hints for the AI...")
        transcription_layout.addWidget(self.prompt_input)
        # Token count label
        self.prompt_token_label = QLabel("")
        self.prompt_token_label.setStyleSheet("color: #555;")
        transcription_layout.addWidget(self.prompt_token_label)
        self.prompt_input.textChanged.connect(self._update_prompt_token_counter)
        # Initial update
        QTimer.singleShot(0, self._update_prompt_token_counter)

        transcription_layout.addStretch()  # Pushes widgets to the top

        # --- Populate Rewording Tab ---
        rephrasing_layout = QVBoxLayout(rephrasing_tab)

        self.rephrasing_checkbox = QCheckBox()
        self.rephrasing_checkbox.setChecked(self.config["rephrasing_enabled"])
        rephrasing_layout.addWidget(self.rephrasing_checkbox)

        self.rephrasing_trigger_label = QLabel()
        rephrasing_layout.addWidget(self.rephrasing_trigger_label)
        self.rephrasing_trigger_word_input = QLineEdit(self.config["rephrasing_trigger_word"])
        rephrasing_layout.addWidget(self.rephrasing_trigger_word_input)

        # New checkbox for context
        self.rephrase_context_checkbox = QCheckBox()
        self.rephrase_context_checkbox.setChecked(self.config["rephrase_use_selection_context"])
        rephrasing_layout.addWidget(self.rephrase_context_checkbox)

        self.rephrasing_prompt_label = QLabel()
        rephrasing_layout.addWidget(self.rephrasing_prompt_label)
        self.rephrasing_prompt_input = QTextEdit(self.config["rephrasing_prompt"])
        rephrasing_layout.addWidget(self.rephrasing_prompt_input)

        self.rephrasing_api_url_label = QLabel()
        rephrasing_layout.addWidget(self.rephrasing_api_url_label)
        self.rephrasing_api_url_input = QLineEdit(self.config["rephrasing_api_url"])
        rephrasing_layout.addWidget(self.rephrasing_api_url_input)

        self.rephrasing_api_key_label = QLabel()
        rephrasing_layout.addWidget(self.rephrasing_api_key_label)
        self.rephrasing_api_key_input = QLineEdit(self.config["rephrasing_api_key"])
        self.rephrasing_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        rephrasing_layout.addWidget(self.rephrasing_api_key_input)

        self.rephrasing_model_label = QLabel()
        rephrasing_layout.addWidget(self.rephrasing_model_label)
        self.rephrasing_model_input = QLineEdit(self.config["rephrasing_model"])
        rephrasing_layout.addWidget(self.rephrasing_model_input)

        # --- Rephrasing Temperature Slider ---
        self.rephrasing_temp_label_title = QLabel()
        rephrasing_layout.addWidget(self.rephrasing_temp_label_title)
        rephrasing_temp_layout = QHBoxLayout()
        self.rephrasing_temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.rephrasing_temp_slider.setRange(0, 100)
        self.rephrasing_temp_slider.setValue(int(self.config["rephrasing_temperature"] * 100))
        self.rephrasing_temp_label = QLabel(f"{self.config['rephrasing_temperature']:.2f}")
        self.rephrasing_temp_slider.valueChanged.connect(self._update_rephrasing_temp_label)
        rephrasing_temp_layout.addWidget(self.rephrasing_temp_slider)
        rephrasing_temp_layout.addWidget(self.rephrasing_temp_label)
        rephrasing_layout.addLayout(rephrasing_temp_layout)

        rephrasing_layout.addStretch()  # Pushes widgets to the top

        # --- Populate Post Rewording Tab ---
        # Re-implemented as split view (list + editor)
        self.max_post_rephrasing_entries = 10
        pr_layout = QVBoxLayout(post_rephrasing_tab)
        self.transformations_info_label = QLabel()
        self.transformations_info_label.setWordWrap(True)
        pr_layout.addWidget(self.transformations_info_label)

        splitter = QSplitter()
        pr_layout.addWidget(splitter, 1)

        # Left list
        # Replace simple QListWidget with drag-enabled one
        class _PostRPList(QListWidget):
            def __init__(self, outer):
                super().__init__()
                self._outer = outer
                self.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
                self.setDragEnabled(True)
                self.setAcceptDrops(True)
                self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
                self.setDefaultDropAction(Qt.DropAction.MoveAction)

            def dropEvent(self, event):
                super().dropEvent(event)
                # After visual reorder, sync underlying data list
                if hasattr(self._outer, '_sync_post_rp_data_from_list'):
                    self._outer._sync_post_rp_data_from_list()

        self.post_rp_list = _PostRPList(self)
        splitter.addWidget(self.post_rp_list)

        # Right editor container
        self.post_rp_editor_container = QWidget()
        editor_layout = QVBoxLayout(self.post_rp_editor_container)
        self.caption_label = QLabel()
        editor_layout.addWidget(self.caption_label)
        self.post_rp_caption_edit = QLineEdit()
        editor_layout.addWidget(self.post_rp_caption_edit)
        self.text_label = QLabel()
        editor_layout.addWidget(self.text_label)
        self.post_rp_text_edit = QTextEdit()
        editor_layout.addWidget(self.post_rp_text_edit, 1)
        splitter.addWidget(self.post_rp_editor_container)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Buttons row
        btn_row = QHBoxLayout()
        self.post_rp_add_btn = QPushButton()
        self.post_rp_remove_btn = QPushButton()
        btn_row.addWidget(self.post_rp_add_btn)
        btn_row.addWidget(self.post_rp_remove_btn)
        btn_row.addStretch(1)
        pr_layout.addLayout(btn_row)

        # Data list
        self.post_rephrasing_data: List[Dict[str, str]] = self.config["post_rephrasing_entries"][:self.max_post_rephrasing_entries]
        self._post_rp_updating = False
        self._load_post_rp_entries_into_list()
        self.post_rp_list.currentRowChanged.connect(self._on_post_rp_selection_changed)
        self.post_rp_caption_edit.textChanged.connect(self._on_post_rp_caption_changed)
        self.post_rp_text_edit.textChanged.connect(self._on_post_rp_text_changed)
        self.post_rp_add_btn.clicked.connect(self._on_post_rp_add_clicked)
        self.post_rp_remove_btn.clicked.connect(self._on_post_rp_remove_clicked)
        self._update_post_rp_ui_state()
        # Select first by default if exists
        if self.post_rp_list.count() > 0:
            self.post_rp_list.setCurrentRow(0)

        # --- Populate General Tab ---
        general_layout = QVBoxLayout(general_tab)

        # UI Language Selector
        self.ui_language_label = QLabel()
        general_layout.addWidget(self.ui_language_label)
        self.ui_language_selector = QComboBox()
        self.ui_language_selector.addItems(["English", "Deutsch", "Español", "Français"]) # Add more as json files are created
        # Set current language
        lang_map = {"en": "English", "de": "Deutsch", "es": "Español", "fr": "Français"}
        current_lang_name = lang_map.get(self.config.get("ui_language", "en"), "English")
        self.ui_language_selector.setCurrentText(current_lang_name)
        self.ui_language_selector.currentTextChanged.connect(self.change_language)
        general_layout.addWidget(self.ui_language_selector)

        self.restore_clipboard_checkbox = QCheckBox()
        self.restore_clipboard_checkbox.setChecked(self.config["restore_clipboard"])
        general_layout.addWidget(self.restore_clipboard_checkbox)

        self.debug_logging_checkbox = QCheckBox()
        self.debug_logging_checkbox.setChecked(self.config["debug_logging"])
        general_layout.addWidget(self.debug_logging_checkbox)

        # New checkbox for file logging
        self.file_logging_checkbox = QCheckBox()
        self.file_logging_checkbox.setChecked(self.config["file_logging"])
        general_layout.addWidget(self.file_logging_checkbox)

        self.play_g_button = QPushButton()
        self.play_g_button.clicked.connect(self.play_latest_recording)
        general_layout.addWidget(self.play_g_button, alignment=Qt.AlignmentFlag.AlignLeft)

        general_layout.addStretch()  # Pushes widgets to the top

        # --- Save Button (outside tabs) ---
        self.save_button = QPushButton()
        self.save_button.clicked.connect(self.save_settings)
        main_layout.addWidget(self.save_button)

        # --- FIX ---
        # The following line is removed. Calling setLayout on a widget that
        # already has a layout (set via the QVBoxLayout(self) constructor)
        # can lead to crashes.
        # self.setLayout(main_layout)

        # Make the settings window 50% wider than default
        self.resize(int(self.sizeHint().width() * 1.28), self.sizeHint().height())

        # Set window icon
        icon_path = resource_path("ressources/app_icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        else:
            self.setWindowIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon))

        # Set all translatable texts
        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        """Updates all UI texts to the currently selected language."""
        # Window Title
        self.setWindowTitle(self.translator.tr("window_title"))

        # Tabs
        self.tabs.setTabText(0, self.translator.tr("tab_transcription"))
        self.tabs.setTabText(1, self.translator.tr("tab_rephrase"))
        self.tabs.setTabText(2, self.translator.tr("tab_transformations"))
        self.tabs.setTabText(3, self.translator.tr("tab_general"))
        self.tabs.setTabToolTip(0, self.translator.tr("tooltip_tab_transcription"))
        self.tabs.setTabToolTip(1, self.translator.tr("tooltip_tab_rephrase"))
        self.tabs.setTabToolTip(2, self.translator.tr("tooltip_tab_transformations"))
        self.tabs.setTabToolTip(3, self.translator.tr("tooltip_tab_general"))

        # Transcription Tab
        self.api_key_label.setText(self.translator.tr("api_key_label"))
        self.api_endpoint_label.setText(self.translator.tr("api_endpoint_label"))
        self.openai_button.setText(self.translator.tr("openai_button"))
        self.groq_button.setText(self.translator.tr("groq_button"))
        self.model_label.setText(self.translator.tr("model_label"))
        self.model_input.setPlaceholderText(self.translator.tr("custom_model_placeholder"))
        self.transcription_temp_label_title.setText(self.translator.tr("temperature_label"))
        self.hotkey_label.setText(self.translator.tr("hotkey_label"))
        self.set_hotkey_button.setText(self.translator.tr("set_hotkey_button"))
        self.input_language_label.setText(self.translator.tr("input_language_label"))
        self.gain_label.setText(self.translator.tr("gain_label"))
        self.transcription_prompt_label.setText(self.translator.tr("transcription_prompt_label"))

        # Rephrase Tab
        self.rephrasing_checkbox.setText(self.translator.tr("rephrase_enable_checkbox"))
        self.rephrasing_checkbox.setToolTip(self.translator.tr("rephrase_enable_tooltip"))
        self.rephrasing_trigger_label.setText(self.translator.tr("rephrase_trigger_label"))
        self.rephrasing_trigger_word_input.setToolTip(self.translator.tr("rephrase_trigger_tooltip"))
        self.rephrase_context_checkbox.setText(self.translator.tr("rephrase_context_checkbox"))
        self.rephrase_context_checkbox.setToolTip(self.translator.tr("rephrase_context_tooltip"))
        self.rephrasing_prompt_label.setText(self.translator.tr("rephrase_prompt_label"))
        self.rephrasing_prompt_input.setPlaceholderText(self.translator.tr("rephrase_prompt_placeholder"))
        self.rephrasing_api_url_label.setText(self.translator.tr("rephrase_api_url_label"))
        self.rephrasing_api_key_label.setText(self.translator.tr("rephrase_api_key_label"))
        self.rephrasing_model_label.setText(self.translator.tr("rephrase_model_label"))
        self.rephrasing_temp_label_title.setText(self.translator.tr("temperature_label"))

        # Transformations Tab
        self.transformations_info_label.setText(self.translator.tr("transformations_info", max_entries=self.max_post_rephrasing_entries))
        self.caption_label.setText(self.translator.tr("caption_label"))
        self.text_label.setText(self.translator.tr("text_label"))
        self.post_rp_text_edit.setPlaceholderText(self.translator.tr("text_placeholder"))
        self.post_rp_add_btn.setText(self.translator.tr("add_button"))
        self.post_rp_remove_btn.setText(self.translator.tr("remove_button"))

        # General Tab
        self.ui_language_label.setText(self.translator.tr("ui_language_label"))
        self.restore_clipboard_checkbox.setText(self.translator.tr("restore_clipboard_checkbox"))
        self.debug_logging_checkbox.setText(self.translator.tr("debug_logging_checkbox"))
        self.file_logging_checkbox.setText(self.translator.tr("file_logging_checkbox"))
        self.play_g_button.setText(self.translator.tr("play_last_recording_button"))
        self.play_g_button.setToolTip(self.translator.tr("play_last_recording_tooltip"))

        # Save Button
        self.save_button.setText(self.translator.tr("save_button"))

        # Update other dynamic texts
        self._update_prompt_token_counter()
        self.init_tray_icon() # Re-init to update menu item texts

    def change_language(self, lang_name: str) -> None:
        """
        Changes the application's UI language.

        Args:
            lang_name (str): The display name of the language to switch to.
        """
        lang_map = {"English": "en", "Deutsch": "de", "Español": "es", "Français": "fr"}
        lang_code = lang_map.get(lang_name, "en")
        self.translator.set_language(lang_code)
        self.config["ui_language"] = lang_code
        self.retranslate_ui()

    def _update_transcription_temp_label(self, value: int) -> None:
        """Updates the label for the transcription temperature slider."""
        self.transcription_temp_label.setText(f"{value / 100.0:.2f}")

    def _update_rephrasing_temp_label(self, value: int) -> None:
        """Updates the label for the rephrasing temperature slider."""
        self.rephrasing_temp_label.setText(f"{value / 100.0:.2f}")

    def start_hotkey_capture(self) -> None:
        """Initiates the process of listening for a new hotkey."""
        self.set_hotkey_button.setText(self.translator.tr("hotkey_listening_button"))
        self.set_hotkey_button.setEnabled(False)
        self.captured_keys = set()
        self.hotkey_capture_listener = keyboard.Listener(on_press=self.on_press_capture,
                                                         on_release=self.on_release_capture)
        self.hotkey_capture_listener.start()

    def on_press_capture(self, key: Any) -> None:
        """
        Callback for when a key is pressed during hotkey capture.

        Args:
            key (Any): The key that was pressed.
        """
        self.captured_keys.add(key)
        self.hotkey_display.setText(self.keys_to_string(self.captured_keys))

    def on_release_capture(self, key: Any) -> None:
        """
        Callback for when a key is released, finalizing the hotkey capture.

        Args:
            key (Any): The key that was released.
        """
        if self.hotkey_capture_listener:
            self.hotkey_capture_listener.stop()
            self.hotkey_capture_listener = None
        self.new_hotkey_str = self.keys_to_string(self.captured_keys)
        self.hotkey_display.setText(self.new_hotkey_str)
        self.set_hotkey_button.setText(self.translator.tr("set_hotkey_button"))
        self.set_hotkey_button.setEnabled(True)

    def keys_to_string(self, keys: Set[Any]) -> str:
        """
        Converts a set of pynput key objects to a display string.

        Args:
            keys (Set[Any]): A set of pynput key objects.

        Returns:
            str: The string representation of the hotkey combination.
        """
        key_parts = []
        # CORRECTED: Simplified the sorting key to be more robust.
        for key in sorted(keys, key=str):
            key_name = None
            if isinstance(key, keyboard.Key):
                # For special keys, format them like <key_name>
                key_name = f"<{key.name}>"
            elif isinstance(key, keyboard.KeyCode) and key.char:
                # For regular character keys, just use the character
                key_name = key.char

            if key_name and key_name not in key_parts:
                key_parts.append(key_name)
        return '+'.join(key_parts)

    def string_to_keyset(self, hotkey_str: str) -> Set[Any]:
        """
        Converts a hotkey string from the config back into a set of pynput key objects.

        Args:
            hotkey_str (str): The hotkey string (e.g., "<ctrl>+s").

        Returns:
            Set[Any]: A set of pynput key objects.
        """
        if not hotkey_str:
            return set()

        keyset = set()
        parts = hotkey_str.split('+')
        for part in parts:
            part = part.strip()
            if part.startswith('<') and part.endswith('>'):
                key_name = part[1:-1]
                try:
                    # Attempt to find the key in pynput's Key enum
                    keyset.add(keyboard.Key[key_name])
                except KeyError:
                    logging.error(f"Could not find special key '{key_name}' in pynput library.")
            elif len(part) == 1:
                # It's a regular character key
                keyset.add(keyboard.KeyCode.from_char(part))
        return keyset

    def init_tray_icon(self) -> None:
        """Initializes the system tray icon and its context menu."""
        if hasattr(self, 'tray_icon'):
            self.tray_icon.hide() # Hide old one if exists
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

        tray_menu = QMenu()

        show_action = tray_menu.addAction(self.translator.tr("tray_settings_action"))
        show_action.triggered.connect(self.show)

        copy_action = tray_menu.addAction(self.translator.tr("tray_copy_action"))
        copy_action.triggered.connect(self.copy_last_transcription_to_clipboard)

        # Add "Play Last Recording" action
        self.play_action = tray_menu.addAction(self.translator.tr("tray_play_action"))
        self.play_action.triggered.connect(self.play_latest_recording)
        self.play_action.setEnabled(False)  # Disabled until a recording exists

        # Add "Open Log File" action
        self.open_log_action = tray_menu.addAction(self.translator.tr("tray_log_action"))
        self.open_log_action.triggered.connect(self.open_log_file)
        # Will be enabled/disabled based on log file existence

        # Create "Cancel Recording" action
        self.cancel_action = QAction(self.translator.tr("tray_cancel_action"), self)
        self.cancel_action.triggered.connect(self.cancel_recording)
        self.cancel_action.setVisible(False)  # Hide initially
        tray_menu.addAction(self.cancel_action)

        tray_menu.addSeparator()

        quit_action = tray_menu.addAction(self.translator.tr("tray_quit_action"))
        quit_action.triggered.connect(self.quit_app)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

        # Initial state update for log file action
        self.update_logfile_menu_action()

    def update_logfile_menu_action(self) -> None:
        """Updates the enabled/disabled state of the 'Open Log File' action in the tray menu."""
        if hasattr(self, 'open_log_action'):
            log_file_path = os.path.join(APP_BASE_DIR, "voice_transcriber.log")
            log_file_exists = os.path.isfile(log_file_path)
            self.open_log_action.setEnabled(log_file_exists)

    def open_log_file(self) -> None:
        """Opens the log file with the system's default application (text editor)."""
        log_file_path = os.path.join(APP_BASE_DIR, "voice_transcriber.log")

        if not os.path.isfile(log_file_path):
            self.show_tray_balloon(self.translator.tr("log_file_not_exist_message"), 2000)
            self.update_logfile_menu_action()  # Update menu state
            return

        try:
            if is_WINDOWS:
                os.startfile(log_file_path)
            elif is_MACOS:
                subprocess.call(['open', log_file_path])
            else:
                subprocess.call(['xdg-open', log_file_path])
        except Exception as e:
            logging.error(f"Failed to open log file: {e}")
            self.show_tray_balloon(self.translator.tr("log_file_open_fail_message", error=e), 3000)

    def init_manual_hotkey_listener(self) -> None:
        """Initializes the manual, low-level keyboard listener."""
        # This set will hold the keys for our desired hotkey combination
        self.target_hotkey_set = self.string_to_keyset(self.hotkey_str)

        # This set tracks which keys are currently held down
        self.pressed_keys = set()

        # This flag prevents the hotkey from firing repeatedly while held down
        self.hotkey_fired = False

        # Stop any previous listeners if this is called again
        if hasattr(self, 'manual_listener') and self.manual_listener.is_alive():
            self.manual_listener.stop()

        if not self.target_hotkey_set:
            logging.warning("No valid hotkey set. Hotkey listener will not start.")
            return

        # Create and start the new listener
        self.manual_listener = keyboard.Listener(
            on_press=self._on_hotkey_press,
            on_release=self._on_hotkey_release
        )
        self.manual_listener.start()
        logging.info(f"Manual hotkey listener started for combo: {self.hotkey_str}")

    def _on_hotkey_press(self, key: Any) -> None:
        """
        Callback for the manual listener when any key is pressed.

        Args:
            key (Any): The key that was pressed.
        """
        # Return early if the target hotkey is empty to avoid accidental triggers
        if not self.target_hotkey_set:
            return

        self.pressed_keys.add(key)

        # Check if all target keys are now in the set of pressed keys
        if self.target_hotkey_set.issubset(self.pressed_keys):
            # Fire the event only once per press-down sequence
            if not self.hotkey_fired:
                self.hotkey_fired = True
                logging.info(f"Manual hotkey combo detected: {self.hotkey_str}")
                self.toggle_recording()

                # --- FIX ---
                # Clear the set of pressed keys immediately after a successful trigger.
                # This prevents stale/missed key-up events (especially for modifier keys
                # like Ctrl) from causing incorrect future activations.
                self.pressed_keys.clear()

    def _on_hotkey_release(self, key: Any) -> None:
        """
        Callback for the manual listener when any key is released.

        Args:
            key (Any): The key that was released.
        """
        # If any of our hotkey keys are released, we can fire the event again next time.
        if key in self.target_hotkey_set:
            self.hotkey_fired = False

        # Remove the key from the set of pressed keys
        if key in self.pressed_keys:
            self.pressed_keys.remove(key)

    def preload_sounds(self) -> None:
        """Load small WAV files fully into memory for instant playback."""
        sound_names = ["sound_start.wav", "sound_end.wav"]
        if not self.pyaudio_instance:
            try:
                self.pyaudio_instance = pyaudio.PyAudio()
            except Exception as e:
                logging.warning(f"PyAudio init failed (sounds disabled): {e}")
                return
        for name in sound_names:
            path = resource_path(os.path.join("ressources", name))
            if not os.path.isfile(path):
                logging.debug(f"Sound file missing (skip preload): {path}")
                continue
            try:
                with wave.open(path, 'rb') as wf:
                    sampwidth = wf.getsampwidth()
                    channels = wf.getnchannels()
                    rate = wf.getframerate()
                    frames = wf.readframes(wf.getnframes())
                    self.sound_cache[name] = (sampwidth, channels, rate, frames)
                    logging.debug(f"Preloaded sound {name} (ch={channels}, rate={rate})")
            except Exception as e:
                logging.warning(f"Failed to preload {name}: {e}")

    def _preopen_streams(self) -> None:
        """Iterate through cached sounds and ensure an output stream is open for each format."""
        logging.debug("Pre-opening audio streams for cached sounds...")
        for sound_data in self.sound_cache.values():
            sampwidth, channels, rate, _ = sound_data
            try:
                # This will get an existing stream or create and cache a new one
                self._get_output_stream(sampwidth, channels, rate)
            except Exception as e:
                logging.error(f"Failed to pre-open stream for format ({sampwidth}, {channels}, {rate}): {e}")
        logging.debug("Finished pre-opening streams.")

    def _get_output_stream(self, sampwidth: int, channels: int, rate: int):
        """
        Get or create a reusable PyAudio output stream keyed by format.

        Args:
            sampwidth (int): The sample width in bytes.
            channels (int): The number of audio channels.
            rate (int): The sampling rate in Hz.

        Returns:
            A PyAudio stream object or None if an error occurred.
        """
        key = (sampwidth, channels, rate)
        if key in self._output_streams:
            return self._output_streams[key]
        if not self.pyaudio_instance:
            try:
                self.pyaudio_instance = pyaudio.PyAudio()
            except Exception as e:
                logging.error(f"PyAudio init failed for playback: {e}")
                return None
        try:
            stream = self.pyaudio_instance.open(
                format=self.pyaudio_instance.get_format_from_width(sampwidth),
                channels=channels,
                rate=rate,
                output=True
            )
            self._output_streams[key] = stream
            return stream
        except Exception as e:
            logging.error(f"Could not open output stream: {e}")
            return None

    def play_sound(self, filename: str) -> None:
        """
        Low-latency playback of preloaded short WAV. Falls back to on-demand load.

        Args:
            filename (str): The name of the sound file to play.
        """

        def _play_cached():
            data_tuple = self.sound_cache.get(filename)
            if not data_tuple:
                # Fallback: attempt one-off load (slower)
                path = resource_path(filename)
                if not os.path.isfile(path):
                    logging.debug(f"Sound file not found: {path}")
                    return
                try:
                    with wave.open(path, 'rb') as wf:
                        sampwidth = wf.getsampwidth()
                        channels = wf.getnchannels()
                        rate = wf.getframerate()
                        frames = wf.readframes(wf.getnframes())
                        data_tuple = (sampwidth, channels, rate, frames)
                        # Cache for future
                        self.sound_cache[filename] = data_tuple
                except Exception as e:
                    logging.error(f"Failed to load sound '{filename}': {e}")
                    return
            sampwidth, channels, rate, frames = data_tuple
            stream = self._get_output_stream(sampwidth, channels, rate)
            if not stream:
                return
            try:
                stream.write(frames)
            except Exception as e:
                logging.error(f"Playback error for '{filename}': {e}")

        # Very short sounds: writing directly is fine; keep daemon thread to not block caller thread
        threading.Thread(target=_play_cached, daemon=True).start()

    def toggle_recording(self) -> None:
        """Toggles the audio recording state."""
        if self.is_recording:
            self.is_recording = False
            self.cancel_action.setVisible(False)  # Hide cancel option
            self.tray_icon.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
            self.show_tray_balloon(self.translator.tr("recording_stopped_message"), 2000)
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join()
            logging.info("Recording stopped. Processing audio.")
            self.play_sound('ressources/sound_end.wav')
            self.process_recording()
        else:
            # Reset context for the new operation
            self.current_transcription_context = ""
            # Check if context from selection should be added at the start
            if self.config["rephrase_use_selection_context"]:
                context_text = self.get_selected_text()
                if context_text:
                    self.current_transcription_context = context_text
                    logging.info(f"Captured context for rephrasing: {context_text}")

            self.is_recording = True
            self.cancel_action.setVisible(True)  # Show cancel option
            self.recorded_frames = []
            self.tray_icon.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
            # Start audio capture quickly
            self.recording_thread = threading.Thread(target=self.record_audio, daemon=True)
            self.recording_thread.start()
            # Play start sound immediately (removed artificial sleep)
            self.play_sound('ressources/sound_start.wav')
            self.show_tray_balloon(self.translator.tr("recording_running_message"), 99999999)
            logging.info("Recording started.")

    def cancel_recording(self) -> None:
        """Stops the current recording without processing it."""
        if not self.is_recording:
            return

        logging.info("Recording canceled by user.")
        self.is_recording = False
        self.cancel_action.setVisible(False)  # Hide the action again

        # Wait for the recording thread to finish cleanly
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join()

        # Reset UI and provide feedback
        self.tray_icon.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.show_tray_balloon(self.translator.tr("recording_canceled_message"), 2000)
        self.play_sound('ressources/sound_end.wav')

    def record_audio(self) -> None:
        """Records audio using PyAudio (16-bit mono)."""
        logging.info("Audio recording thread started (PyAudio).")
        try:
            if not self.pyaudio_instance:
                self.pyaudio_instance = pyaudio.PyAudio()
            stream = self.pyaudio_instance.open(format=pyaudio.paInt16,
                                                channels=1,
                                                rate=self.samplerate,
                                                input=True,
                                                frames_per_buffer=self.chunk_size)
        except Exception as e:
            logging.error(f"Could not open audio input stream: {e}")
            self.is_recording = False
            return
        while self.is_recording:
            try:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                self.recorded_frames.append(data)
            except Exception as e:
                logging.error(f"Error while reading audio stream: {e}")
                break
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        logging.info("Audio recording thread finished.")

    def _apply_gain_inplace(self, raw_audio: bytearray, gain_db: float) -> None:
        """
        Apply gain (dB) to 16-bit PCM mono audio in-place without numpy.

        Args:
            raw_audio (bytearray): The raw audio data as a bytearray.
            gain_db (float): The gain to apply in decibels.
        """
        if gain_db <= 0:
            return
        factor = math.pow(10.0, gain_db / 20.0)
        for i in range(0, len(raw_audio), 2):
            sample = struct.unpack_from('<h', raw_audio, i)[0]
            amplified = int(sample * factor)
            if amplified > 32767:
                amplified = 32767
            elif amplified < -32768:
                amplified = -32768
            struct.pack_into('<h', raw_audio, i, amplified)

    def process_recording(self) -> None:
        """Processes the recorded audio, saves it to a file, and starts transcription."""
        if not self.recorded_frames:
            logging.warning("No audio data was recorded.")
            self.show_tray_balloon(self.translator.tr("no_audio_captured_message"), 2000)
            return
        temp_dir: str = tempfile.gettempdir()
        filename: str = f"pyvoicetranscriber_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        filepath: str = os.path.join(temp_dir, filename)
        raw_audio = b''.join(self.recorded_frames)
        audio_bytes = bytearray(raw_audio)
        gain_db = float(self.config["gain_db"])
        if gain_db > 0:
            try:
                self._apply_gain_inplace(audio_bytes, gain_db)
                logging.info(f"Applied +{gain_db} dB gain to recording.")
            except Exception as e:
                logging.error(f"Gain application failed: {e}")
        try:
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.samplerate)
                wf.writeframes(audio_bytes)
            logging.info(f"Recording saved to: {filepath}")
            # Enable the play action in the tray menu now that a file exists
            if hasattr(self, 'play_action'):
                self.play_action.setEnabled(True)
        except Exception as e:
            logging.error(f"Failed to write WAV file: {e}")
            self.show_tray_balloon("Failed to save audio.", 3000)
            return
        self.keep_only_latest_recording(filepath)
        self.start_transcription_worker(filepath)

    def cleanup_old_recordings(self):
        """Delete all old pyvoicetranscriber_recording_*.wav files on startup."""
        temp_dir = tempfile.gettempdir()
        for fname in os.listdir(temp_dir):
            if fname.startswith("pyvoicetranscriber_recording_") and fname.endswith(".wav"):
                try:
                    os.remove(os.path.join(temp_dir, fname))
                except Exception as e:
                    logging.warning(f"Could not delete old recording {fname}: {e}")

    def keep_only_latest_recording(self, latest_path):
        """
        Delete all but the latest recording file after a new recording is saved.

        Args:
            latest_path (str): The path to the latest recording file to keep.
        """
        temp_dir = tempfile.gettempdir()
        # Find all matching files
        files = [
            os.path.join(temp_dir, f)
            for f in os.listdir(temp_dir)
            if f.startswith("pyvoicetranscriber_recording_") and f.endswith(".wav")
        ]
        if not files:
            return
        # Sort by modification date (newest first)
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        # Keep only the newest file
        for f in files[1:]:
            try:
                os.remove(f)
            except Exception as e:
                logging.warning(f"Could not delete old recording {os.path.basename(f)}: {e}")

    def play_latest_recording(self):
        """Open the latest recording in the system's default media player."""
        temp_dir = tempfile.gettempdir()
        files = [f for f in os.listdir(temp_dir) if
                 f.startswith("pyvoicetranscriber_recording_") and f.endswith(".wav")]
        if not files:
            self.show_tray_balloon(self.translator.tr("no_recording_found_message"), 2000)
            return
        files.sort(reverse=True)
        latest = os.path.join(temp_dir, files[0])
        try:
            if is_WINDOWS:
                os.startfile(latest)
            elif is_MACOS:
                subprocess.call(['open', latest])
            else:
                subprocess.call(['xdg-open', latest])
        except Exception as e:
            self.show_tray_balloon(self.translator.tr("could_not_play_file_message", error=e), 2000)

    def start_transcription_worker(self, audio_path: str) -> None:
        """Creates and starts a new thread for the transcription worker."""
        self.show_tray_balloon(self.translator.tr("transcription_progress_message"), 3000)
        thread = QThread()
        # The config now stores the language code directly
        lang_code = self.config["input_language"]

        worker = TranscriptionWorker(
            api_key=self.config["api_key"], api_endpoint=self.config["api_endpoint"],
            audio_path=audio_path, prompt=self.config["prompt"],
            model=self.config["model"], language=lang_code,
            temperature=self.config["transcription_temperature"]
        )
        worker.moveToThread(thread)
        self.active_workers.append(worker)
        self.active_threads.append(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self.on_transcription_finished)
        worker.error.connect(self.on_transcription_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda t=thread, w=worker: self._cleanup_worker(t, w))
        thread.start()

    def _cleanup_worker(self, thread: QThread, worker: TranscriptionWorker) -> None:
        """
        Removes finished worker/thread references to allow garbage collection.

        Args:
            thread (QThread): The QThread that has finished.
            worker (TranscriptionWorker): The worker that has finished.
        """
        if worker in self.active_workers: self.active_workers.remove(worker)
        if thread in self.active_threads: self.active_threads.remove(thread)
        logging.info("Worker thread cleaned up.")

    def on_transcription_finished(self, text: str) -> None:
        """
        Handles the successful transcription result.

        Args:
            text (str): The transcribed text.
        """
        processed = text.strip('"\'“”‘’ ')
        prompt = self.config["prompt"].strip()
        if not processed or (prompt and processed.lower() == prompt.lower()):
            self.show_tray_balloon(self.translator.tr("no_speech_recognized_message"), 2000)
            logging.info("Transcription result was empty or matched the prompt, ignoring.")
            return

        # Rewording/Rephrasing via AI if activated
        if self.config["rephrasing_enabled"]:
            trigger_words_str = self.config["rephrasing_trigger_word"].lower()
            trigger_words = [word.strip() for word in trigger_words_str.split(',') if word.strip()]

            should_rephrase = False
            if not trigger_words:
                # If trigger list is empty, always rephrase when enabled
                should_rephrase = True
            else:
                # If any trigger word is found, rephrase
                processed_lower = processed.lower()
                if any(trigger in processed_lower for trigger in trigger_words):
                    should_rephrase = True

            if should_rephrase:
                try:
                    self.show_tray_balloon(self.translator.tr("rephrasing_transcript_message", processed_text=processed), 3000)

                    rephrased = self.rephrase_text_with_gpt(
                        processed,
                        self.config["rephrasing_prompt"],
                        self.config["rephrasing_api_url"],
                        self.config["rephrasing_api_key"],
                        self.config["rephrasing_model"],
                        self.config["rephrasing_temperature"],
                        context=self.current_transcription_context
                    )
                    if rephrased:
                        processed = rephrased
                except Exception as e:
                    logging.error(f"Rephrasing failed: {e}")
                    self.show_tray_balloon(self.translator.tr("rephrasing_failed_message", error=e), 3000)

        self.last_transcription = processed
        self.insert_transcribed_text(self.last_transcription)

    def rephrase_text_with_gpt(self, text: str, prompt: str, api_url: str, api_key: str, model: str, temperature: float, context: str = "") -> str:
        """
        Sends the User's text to the rephrasing API and returns the rephrased text.

        Args:
            text (str): The text to be rephrased.
            prompt (str): The system prompt to guide the rephrasing model.
            api_url (str): The URL of the rephrasing API.
            api_key (str): The API key for the rephrasing service.
            model (str): The name of the language model to use for rephrasing.
            temperature (float): The sampling temperature for the model.
            context (str, optional): Additional context (e.g., selected text). Defaults to "".

        Returns:
            str: The rephrased text.

        Raises:
            ValueError: If API settings are incomplete.
            Exception: If the API request fails.
        """
        if not api_url or not api_key or not model:
            raise ValueError("Rephrasing API settings are incomplete.")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        messages = []
        if prompt and 0:
            messages.append({"role": "system", "content": prompt})

        user_content = f"{prompt}\n\nText: {text}"
        if context:
            user_content = f"{user_content}\n\nContext: {context}"

        messages.append({"role": "user", "content": user_content})
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        logging.debug(f"Rephrasing request data: {data}")
        try:
            response = requests.post(api_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
        except Exception as e:
            raise Exception(f"Rephrasing API request failed: {e}")
        result = response.json()

        # OpenAI/Groq style: result['choices'][0]['message']['content']
        res = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        logging.info(f"Rephrasing result: {res}")
        return res

    def on_transcription_error(self, error_message: str, audio_file_path: str) -> None:
        """
        Handles errors that occur during transcription.

        Args:
            error_message (str): The error message.
            audio_file_path (str): The path to the audio file that caused the error.
        """
        logging.error(f"Transcription error: {error_message}")
        self.show_tray_balloon(self.translator.tr("transcription_failed_message"), 4000)
        self.tray_icon.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))  # Reset icon

        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Warning)
        msg_box.setText(self.translator.tr("transcription_error_title"))
        msg_box.setInformativeText(self.translator.tr("transcription_error_text", error_message=error_message, audio_file_path=audio_file_path))
        msg_box.setWindowTitle(self.translator.tr("transcription_error_title"))
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Retry)
        if msg_box.exec() == QMessageBox.StandardButton.Retry:
            self.start_transcription_worker(audio_file_path)

    def get_selected_text(self) -> str:
        """
        Retrieves the currently selected text from any application via Pressing Ctrl+C to copy
        to clipboard and then reading it.

        If no text is selected, it would yield the current Clipboard content - this is recognized
        and avoided

        Restores clipboard content if configured.

        Returns:
            str: The selected text, or an empty string if nothing is selected.
        """
        try:
            restore = self.config["restore_clipboard"]
            old_clipboard = copykitten.paste() if restore else None

            # Use Ctrl+C to copy selected text to clipboard
            if is_MACOS:
                pyautogui.hotkey('command', 'c')
            else:
                pyautogui.hotkey('ctrl', 'c')

            time.sleep(0.1)  # Allow time for clipboard to update
            selected_text = copykitten.paste()

            # If no text was selected, we might have copied the current clipboard content
            if not selected_text or selected_text == old_clipboard:
                logging.debug("No new text selected. Context will be empty.")
                selected_text = ""

            if restore and old_clipboard is not None:
                copykitten.copy(old_clipboard)
                logging.debug("Clipboard content restored.")
            return selected_text
        except Exception as e:
            logging.error(f"Failed to retrieve selected text: {e}")
            return ""

    def insert_transcribed_text(self, text: str) -> None:
        """
        Inserts transcribed text, preferably using clipboard for reliability.
        Restore functionality is optional based on user config.

        Args:
            text (str): The text to insert.
        """
        if not text: return
        logging.debug("Inserting transcribed text.")
        try:
            restore = self.config["restore_clipboard"]
            old_clipboard = copykitten.paste() if restore else None

            copykitten.copy(text)

            # Platform-aware paste hotkey
            if is_MACOS:
                pyautogui.hotkey('command', 'v')
            else:
                pyautogui.hotkey('ctrl', 'v')
            time.sleep(0.1)  # Brief pause to ensure paste command is processed

            if restore and old_clipboard is not None:
                copykitten.copy(old_clipboard)
                logging.debug("Clipboard content restored.")
        except Exception as e:
            logging.error(f"Failed to insert text via clipboard: {e}")

    def copy_last_transcription_to_clipboard(self) -> None:
        """Copies the last transcription to the system clipboard."""
        if not self.last_transcription:
            self.show_tray_balloon(self.translator.tr("no_transcription_to_copy_message"), 2000)
            return
        copykitten.copy(self.last_transcription)
        self.show_tray_balloon(self.translator.tr("transcription_copied_message"), 2000)

    def apply_logging_configuration(self) -> None:
        """Apply logging level and file handler based on current config."""
        logger = logging.getLogger()
        # Update level
        logger.setLevel(logging.DEBUG if self.config["debug_logging"] else logging.INFO)
        # Remove existing file handler if present
        if getattr(self, '_file_log_handler', None):
            try:
                logger.removeHandler(self._file_log_handler)
                self._file_log_handler.close()
            except Exception:
                pass
            self._file_log_handler = None

        # Add file handler if enabled
        if self.config["file_logging"]:
            try:
                log_path = os.path.join(APP_BASE_DIR, "voice_transcriber.log")
                fh = logging.FileHandler(log_path, encoding='utf-8')
                fh.setLevel(logging.DEBUG)  # always capture full detail in file
                fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
                logger.addHandler(fh)
                self._file_log_handler = fh
                logging.info(f"File logging enabled: {log_path}")
            except Exception as e:
                logging.error(f"Failed to enable file logging: {e}")

    def _collect_validation_warnings(self, model_raw: str) -> List[str]:
        """Return a list of validation warning strings based on current form values.
        Rules (all case-insensitive):
        - If endpoint contains 'openai': model must contain 'openai' AND key must start with 'sk-'
        - If endpoint contains 'groq': model must contain 'groq' AND key must start with 'gsk'
        (Warnings are hints only; saving proceeds regardless.)
        """
        warnings: List[str] = []
        endpoint = self.api_endpoint_input.text().strip().lower()
        model_lc = model_raw.strip().lower()
        api_key = self.api_key_input.text().strip()
        api_key_lc = api_key.lower()

        if 'openai' in endpoint:
            if 'openai' not in model_lc:
                warnings.append("API endpoint contains 'openai', but the selected model does not contain 'openai'.")
            if not api_key_lc.startswith('sk-'):
                warnings.append("OpenAI API Key should start with 'sk-'.")
        if 'groq' in endpoint:
            if 'groq' not in model_lc:
                warnings.append("API endpoint contains 'groq', but the selected model does not contain 'groq'.")
            if not api_key_lc.startswith('gsk'):
                warnings.append("Groq API Key should start with 'gsk'.")

        # Whisper prompt length validation (soft warning only)
        if 'whisper' in model_lc:
            prompt_txt = self.prompt_input.toPlainText()
            tokens = estimate_tokens(prompt_txt)
            if tokens > WHISPER_PROMPT_TOKEN_LIMIT:
                warnings.append(f"Transcription Prompt exceeds the estimated limited the Whisper model can handle (max {WHISPER_PROMPT_TOKEN_LIMIT} tokens). ")
        return warnings

    # ---------------- Post Rewording (New Implementation) ----------------
    def _load_post_rp_entries_into_list(self) -> None:
        # Preserve current caption for selection restore
        current_row = self.post_rp_list.currentRow()
        current_item_caption = None
        if current_row >= 0:
            itm = self.post_rp_list.item(current_row)
            if itm:
                current_item_caption = itm.text()
        self.post_rp_list.clear()
        for entry in self.post_rephrasing_data:
            caption = entry.get("caption", "").strip() or "(ohne Caption)"
            item = QListWidgetItem(caption)
            # Store reference to dict so we can rebuild ordering after drag&drop
            item.setData(Qt.ItemDataRole.UserRole, entry)
            self.post_rp_list.addItem(item)
        # Try to restore selection
        if current_item_caption:
            for i in range(self.post_rp_list.count()):
                if self.post_rp_list.item(i).text() == current_item_caption:
                    self.post_rp_list.setCurrentRow(i)
                    break

    def _sync_post_rp_data_from_list(self) -> None:
        # Build new list order based on item sequence and stored dict references
        new_list: List[Dict[str, str]] = []
        for i in range(self.post_rp_list.count()):
            item = self.post_rp_list.item(i)
            ref = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(ref, dict):
                new_list.append(ref)
        if len(new_list) == len(self.post_rephrasing_data):
            self.post_rephrasing_data = new_list
        self._update_post_rp_ui_state()

    def _on_post_rp_selection_changed(self, row: int) -> None:
        if row < 0 or row >= len(self.post_rephrasing_data):
            self._post_rp_updating = True
            self.post_rp_caption_edit.clear()
            self.post_rp_text_edit.clear()
            self.post_rp_caption_edit.setEnabled(False)
            self.post_rp_text_edit.setEnabled(False)
            self._post_rp_updating = False
            self._update_post_rp_ui_state()
            return
        self._post_rp_updating = True
        entry = self.post_rephrasing_data[row]
        self.post_rp_caption_edit.setEnabled(True)
        self.post_rp_text_edit.setEnabled(True)
        self.post_rp_caption_edit.setText(entry.get("caption", ""))
        self.post_rp_text_edit.setPlainText(entry.get("text", ""))
        self._post_rp_updating = False
        self._update_post_rp_ui_state()

    def _on_post_rp_caption_changed(self, text: str) -> None:
        if self._post_rp_updating:
            return
        row = self.post_rp_list.currentRow()
        if 0 <= row < len(self.post_rephrasing_data):
            self.post_rephrasing_data[row]["caption"] = text
            item = self.post_rp_list.item(row)
            if item:
                item.setText(text.strip() or "(ohne Caption)")
        self._sync_post_rp_data_from_list()

    def _on_post_rp_text_changed(self) -> None:
        if self._post_rp_updating:
            return
        row = self.post_rp_list.currentRow()
        if 0 <= row < len(self.post_rephrasing_data):
            self.post_rephrasing_data[row]["text"] = self.post_rp_text_edit.toPlainText()

    def _on_post_rp_add_clicked(self) -> None:
        if len(self.post_rephrasing_data) >= self.max_post_rephrasing_entries:
            return
        new_entry = {"caption": "", "text": ""}
        self.post_rephrasing_data.append(new_entry)
        item = QListWidgetItem("(ohne Caption)")
        item.setData(Qt.ItemDataRole.UserRole, new_entry)
        self.post_rp_list.addItem(item)
        self.post_rp_list.setCurrentRow(self.post_rp_list.count() - 1)
        self._update_post_rp_ui_state()

    def _on_post_rp_remove_clicked(self) -> None:
        row = self.post_rp_list.currentRow()
        if 0 <= row < len(self.post_rephrasing_data):
            # Prevent intermediate selection-change handling while mutating
            self._post_rp_updating = True
            self.post_rp_list.blockSignals(True)
            del self.post_rephrasing_data[row]
            self.post_rp_list.takeItem(row)

            # Compute new target row (stay at same index, fallback to last existing)
            if row >= self.post_rp_list.count():
                row = self.post_rp_list.count() - 1

            # Re-select if any remain
            if row >= 0:
                self.post_rp_list.setCurrentRow(row)

            self.post_rp_list.blockSignals(False)
            self._post_rp_updating = False
            # Manually refresh editor state (selection signal suppressed or earlier blanked it)
            self._on_post_rp_selection_changed(row if row >= 0 else -1)
        self._update_post_rp_ui_state()

    def _update_post_rp_ui_state(self) -> None:
        count = len(self.post_rephrasing_data)
        self.post_rp_add_btn.setEnabled(count < self.max_post_rephrasing_entries)
        self.post_rp_remove_btn.setEnabled(count > 0 and self.post_rp_list.currentRow() >= 0)

    def _save_current_post_rp_edits(self) -> None:
        # Data already live-updated via signals; placeholder for future flush logic.
        pass

    def save_settings(self) -> None:
        """Saves settings, restarts the hotkey listener."""
        # Clean model name: remove anything in parentheses and trailing whitespace
        model_raw = self.model_input.text() if self.model_dropdown.currentText() == "Custom" else self.model_dropdown.currentText()
        # Perform validation (warnings only)
        warnings = self._collect_validation_warnings(model_raw)
        if warnings:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setWindowTitle(self.translator.tr("validation_warning_title"))
            msg.setText(self.translator.tr("validation_warning_text"))
            msg.setInformativeText("\n".join(f"- {w}" for w in warnings))
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
        self.config["api_key"] = self.api_key_input.text()
        self.config["api_endpoint"] = self.api_endpoint_input.text()
        self.config["model"] = model_raw
        self.config["transcription_temperature"] = self.transcription_temp_slider.value() / 100.0
        # Save the language code, not the display name
        lang_display_name = self.language_input.currentText()
        self.config["input_language"] = LANGUAGES.get(lang_display_name, "en")
        self.config["prompt"] = self.prompt_input.toPlainText()
        self.config["restore_clipboard"] = self.restore_clipboard_checkbox.isChecked()
        self.config["debug_logging"] = self.debug_logging_checkbox.isChecked()
        self.config["gain_db"] = float(self.gain_input.text() or 0)
        # File logging
        self.config["file_logging"] = self.file_logging_checkbox.isChecked()

        self.config["rephrasing_enabled"] = self.rephrasing_checkbox.isChecked()
        # Get tags from widget and save as comma-separated string
        self.config["rephrasing_trigger_word"] = self.rephrasing_trigger_word_input.text()
        self.config["rephrase_use_selection_context"] = self.rephrase_context_checkbox.isChecked()
        self.config["rephrasing_prompt"] = self.rephrasing_prompt_input.toPlainText()
        self.config["rephrasing_api_url"] = self.rephrasing_api_url_input.text()
        self.config["rephrasing_api_key"] = self.rephrasing_api_key_input.text()
        self.config["rephrasing_model"] = self.rephrasing_model_input.text()
        self.config["rephrasing_temperature"] = self.rephrasing_temp_slider.value() / 100.0
        # Post Rewording entries (new)
        if hasattr(self, 'post_rephrasing_data'):
            self._save_current_post_rp_edits()
            self.config["post_rephrasing_entries"] = self.post_rephrasing_data

        if self.new_hotkey_str and self.new_hotkey_str != self.hotkey_str:
            self.hotkey_str = self.new_hotkey_str
            self.config["hotkey"] = self.new_hotkey_str
            # Re-initialize the listener with the new key set
            self.init_manual_hotkey_listener()
        self.new_hotkey_str = None

        self.save_config()
        self.show_tray_balloon(self.translator.tr("settings_saved_message", hotkey=self.hotkey_str), 2000)
        # Apply logging changes (level + file handler)
        self.apply_logging_configuration()

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Overrides the close event to hide the window instead of quitting.

        Args:
            event (QCloseEvent): The close event.
        """
        event.ignore()
        self.hide()
        #self.show_tray_balloon("Settings hidden. App is still running.", 2000)

        # After closing, delete all old recordings except the last one
        temp_dir = tempfile.gettempdir()
        files = [f for f in os.listdir(temp_dir) if
                 f.startswith("pyvoicetranscriber_recording_") and f.endswith(".wav")]
        if len(files) > 1:
            files.sort(reverse=True)
            for f in files[1:]:
                try:
                    os.remove(os.path.join(temp_dir, f))
                except Exception as e:
                    logging.warning(f"Could not delete old recording {f}: {e}")

    def quit_app(self) -> None:
        """Quits the application cleanly after confirmation."""
        reply = QMessageBox.question(self, self.translator.tr("quit_dialog_title"),
                                     self.translator.tr("quit_dialog_text"),
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.No:
            return

        logging.info("Quitting application.")
        if hasattr(self, 'manual_listener') and self.manual_listener.is_alive():
            self.manual_listener.stop()
        try:
            # Close any output streams
            for s in getattr(self, '_output_streams', {}).values():
                try:
                    s.stop_stream()
                    s.close()
                except Exception:
                    pass
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
        except Exception:
            pass
        try:
            if getattr(self, '_file_log_handler', None):
                logging.getLogger().removeHandler(self._file_log_handler)
                try:
                    self._file_log_handler.close()
                except Exception:
                    pass
                self._file_log_handler = None
        except Exception:
            pass
        self.tray_icon.hide()
        QApplication.instance().quit()

    def _update_prompt_token_counter(self) -> None:
        """Update the token counter label below the transcription prompt."""
        text = self.prompt_input.toPlainText()
        tokens = estimate_tokens(text)
        limit = WHISPER_PROMPT_TOKEN_LIMIT if 'whisper' in (self.model_dropdown.currentText().lower() + ' ' + self.model_input.text().lower()) else None
        if limit:
            over = tokens > limit
            color = 'red' if over else ('#aa7700' if tokens > int(limit*0.85) else '#555')
            exceeded_text = self.translator.tr("token_exceeded_text") if over else ""
            self.prompt_token_label.setText(self.translator.tr("token_counter_exceeded_label", tokens=tokens, limit=limit, exceeded_text=exceeded_text))
            self.prompt_token_label.setStyleSheet(f"color: {color};")
        else:
            self.prompt_token_label.setText(self.translator.tr("token_counter_label", tokens=tokens))
            self.prompt_token_label.setStyleSheet("color: #555;")

# GITHUB COPILOT: KEEP THIS CODE TO MAKE THIS APPLICATION RUNNABLE
# Keep the Main Application Loop to directly run the application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    transcriber_app = VoiceTranscriberApp()
    sys.exit(app.exec())
