"""Recording controls: push-to-talk (below the recording hotkey), warm microphone, length, format."""
from __future__ import annotations

import logging
import threading

from PyQt6.QtCore import QSignalBlocker
from PyQt6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QSpinBox, QWidget

from app.audio.aac_encoder import available_aac_bitrates, warm_up_encoder
from app.core.env import is_WINDOWS
from app.core.input_languages import DETECT_LANGUAGE, MAX_PALETTE_LANGUAGES, palette_languages
from app.ui.language_flags import flag_pixmap
from app.ui.language_names import input_language_items
from app.ui.multi_select_combo import MultiSelectComboBox
from app.ui.settings.base import SettingsWindowBase


class RecordingPage(SettingsWindowBase):
    """Recording controls, including the AAC format offered after a background encoder probe."""

    def _build_recording_controls(self) -> None:
        """Create the recording widgets inside the recording card (language and gain come from the form)."""
        layout = self.recording_group_layout
        # Push-to-talk changes how the recording hotkey works, so it sits below that hotkey.
        self.push_to_talk_checkbox = QCheckBox(self)
        self.hotkey_v_layout.addWidget(self.push_to_talk_checkbox)
        # Refines the gain in the row above, so it opens the recording card's own options.
        self.gain_normalize_checkbox = QCheckBox(self)
        layout.addWidget(self.gain_normalize_checkbox)
        self.play_sounds_checkbox = QCheckBox(self)
        layout.addWidget(self.play_sounds_checkbox)
        self.windows_keep_mic_hot_checkbox = QCheckBox(self)
        layout.addWidget(self.windows_keep_mic_hot_checkbox)

        self.windows_keep_mic_hot_idle_label = QLabel(self)
        self.windows_keep_mic_hot_idle_input = QSpinBox(self)
        self.windows_keep_mic_hot_idle_input.setRange(1, 240)
        self._add_row(self.windows_keep_mic_hot_idle_label, self.windows_keep_mic_hot_idle_input)

        # Shorter recordings are discarded as mis-taps (all platforms).
        self.min_recording_label = QLabel(self)
        self.min_recording_input = QDoubleSpinBox(self)
        self.min_recording_input.setRange(0.0, 10.0)
        self.min_recording_input.setSingleStep(0.1)
        self.min_recording_input.setDecimals(1)
        self.min_recording_input.setSuffix(" s")
        self._add_row(self.min_recording_label, self.min_recording_input)

        self.recording_format_label = QLabel(self)
        self.recording_format_selector = QComboBox(self)
        self.recording_format_selector.setObjectName("recording_format_selector")
        self.recording_format_selector.addItem("WAV (PCM)", "wav")
        self.recording_bitrate_label = QLabel(self)
        self.recording_bitrate_selector = QComboBox(self)
        self.recording_bitrate_selector.setObjectName("recording_bitrate_selector")
        # AAC entries are added by _apply_aac_bitrates once the deferred encoder probe finishes;
        # without the optional PyAV package the whole row stays hidden (WAV only).
        self._aac_bitrates_probed = False
        self._add_row(self.recording_format_label, self.recording_format_selector,
                      self.recording_bitrate_label, self.recording_bitrate_selector)

        # Races a second request over a fresh connection when the first one stalls.
        self.transcription_hedging_checkbox = QCheckBox(self)
        layout.addWidget(self.transcription_hedging_checkbox)
        self.transcription_hedging_wait_label = QLabel(self)
        self.transcription_hedging_wait_input = QDoubleSpinBox(self)
        self.transcription_hedging_wait_input.setRange(0.2, 30.0)
        self.transcription_hedging_wait_input.setSingleStep(0.1)
        self.transcription_hedging_wait_input.setDecimals(1)
        self.transcription_hedging_wait_input.setSuffix(" s")
        self._add_row(self.transcription_hedging_wait_label, self.transcription_hedging_wait_input)

        for name, code in input_language_items():
            self.language_input.addItem(name, code)
        language = self.config["input_language"]
        code = language.lower() if isinstance(language, str) else DETECT_LANGUAGE
        self.language_input.setCurrentIndex(max(0, self.language_input.findData(code)))
        self._build_palette_language_controls()
        self.gain_input.setText(str(self.config["gain_db"]))

    def _build_palette_language_controls(self) -> None:
        """The palette switch and the dropdown of languages the palette offers."""
        self.recording_palette_language_switch_checkbox = QCheckBox(self)
        self.recording_group_layout.addWidget(self.recording_palette_language_switch_checkbox)
        self.recording_palette_languages_label = QLabel(self)
        self.recording_palette_languages_selector = MultiSelectComboBox(self)
        self._add_row(self.recording_palette_languages_label, self.recording_palette_languages_selector)
        for caption, code in input_language_items():
            self.recording_palette_languages_selector.add_choice(caption, code)
        self.recording_palette_languages_selector.set_checked_values(
            palette_languages(self.config["recording_palette_languages"], self._input_language()))
        self.recording_palette_languages_selector.selection_changed.connect(self._update_palette_languages)
        self._update_palette_languages()

    def _add_row(self, *widgets: QWidget) -> None:
        """Append a left-aligned row of widgets to the recording card."""
        row = QHBoxLayout()
        for widget in widgets:
            row.addWidget(widget)
        row.addStretch()
        self.recording_group_layout.addLayout(row)

    def _connect_recording_controls(self) -> None:
        """Live enable/visibility updates once the saved values are loaded."""
        self.recording_palette_language_switch_checkbox.toggled.connect(self._update_palette_language_row)
        self.language_input.currentIndexChanged.connect(self._update_palette_languages)
        self._update_palette_language_row()
        self.windows_keep_mic_hot_checkbox.stateChanged.connect(self._update_windows_keep_mic_hot_ui_state)
        self.recording_format_selector.currentIndexChanged.connect(self._update_recording_format_controls)
        self.transcription_hedging_checkbox.stateChanged.connect(self._update_hedging_ui_state)
        self._update_windows_keep_mic_hot_ui_state()
        self._update_hedging_ui_state()
        self._update_recording_format_controls()

    def _input_language(self) -> str:
        """Read the form's spoken language, including the empty code for detection."""
        language = self.language_input.currentData()
        return language if isinstance(language, str) else "en"

    def _selected_palette_languages(self, current: str) -> list[str]:
        """The ticked languages, with the form's current language always among them."""
        return palette_languages(self.recording_palette_languages_selector.checked_values(), current)

    def _update_palette_languages(self) -> None:
        """Tick the form's input language (it cannot be unticked), stop at the limit and show the summary."""
        selector = self.recording_palette_languages_selector
        current = self._input_language()
        selected = self._selected_palette_languages(current)
        selector.set_checked_values(selected)
        has_room = len(selected) < MAX_PALETTE_LANGUAGES
        selector.set_enabled_values({code: code != current and (code in selected or has_room)
                                     for _caption, code in input_language_items()})
        auto = self.translator.tr("recording_language_auto")
        selector.set_summary([auto if code == DETECT_LANGUAGE else flag_pixmap(code) or code.upper()
                              for code in selected])

    def _update_palette_language_row(self) -> None:
        """Offer the language choice only while the palette switch is on (the row follows the button)."""
        enabled = self.recording_palette_language_switch_checkbox.isChecked()
        self.recording_palette_languages_selector.setEnabled(enabled)
        self.recording_palette_languages_selector.setVisible(enabled)

    def select_input_language(self, code: str) -> None:
        """Reflect a saved palette pick in the form so a later Save keeps the chosen language."""
        index = self.language_input.findData(code)
        if index < 0:
            return
        with QSignalBlocker(self.language_input):
            self.language_input.setCurrentIndex(index)
        self._update_palette_languages()
        self.switch_transcription_prompt_language(code)

    def _retranslate_recording_controls(self) -> None:
        """Translate the automatic choice in both controls and refresh the summary."""
        detect = self.translator.tr("input_language_detect")
        self.language_input.setItemText(self.language_input.findData(DETECT_LANGUAGE), detect)
        self.recording_palette_languages_selector.set_caption(DETECT_LANGUAGE, detect)
        self._update_palette_languages()

    def _update_windows_keep_mic_hot_ui_state(self) -> None:
        """The idle timeout only applies while the warm microphone is enabled."""
        enabled = is_WINDOWS and self.windows_keep_mic_hot_checkbox.isChecked()
        self.windows_keep_mic_hot_idle_label.setEnabled(enabled)
        self.windows_keep_mic_hot_idle_input.setEnabled(enabled)

    def _update_hedging_ui_state(self, *_args: object) -> None:
        """The wait only applies while hedging is enabled."""
        enabled = self.transcription_hedging_checkbox.isChecked()
        self.transcription_hedging_wait_label.setEnabled(enabled)
        self.transcription_hedging_wait_input.setEnabled(enabled)

    def start_aac_bitrate_probe(self) -> None:
        """Probe the optional PyAV AAC encoder in the background; importing PyAV takes noticeable time.

        With AAC selected, the probe also warms the encoder up, so the first recording does not pay
        for loading the codec libraries.
        """
        warm_up_bitrate = (int(self.config.get("recording_aac_bitrate_kbps", 64))
                           if self.config.get("recording_format") == "aac" else None)

        def probe() -> None:
            try:
                bitrates = available_aac_bitrates()
            except Exception as error:
                logging.warning("AAC encoder probe failed: %s", error)
                bitrates = ()
            self.aac_bitrates_ready.emit(bitrates)
            if warm_up_bitrate in bitrates:
                warm_up_encoder(warm_up_bitrate)

        threading.Thread(target=probe, name="AacBitrateProbe", daemon=True).start()

    def _apply_aac_bitrates(self, bitrates: tuple[int, ...]) -> None:
        """Offer AAC once the probe finished and restore the saved format selection."""
        if bitrates:
            self.recording_format_selector.addItem("AAC (M4A)", "aac")
        for bitrate in bitrates:
            self.recording_bitrate_selector.addItem(f"{bitrate} kbit/s", bitrate)
        self.recording_format_selector.setCurrentIndex(max(0, self.recording_format_selector.findData(
            self.config.get("recording_format", "wav"))))
        self.recording_bitrate_selector.setCurrentIndex(max(0, self.recording_bitrate_selector.findData(
            self.config.get("recording_aac_bitrate_kbps", 64))))
        self._aac_bitrates_probed = True
        self._update_recording_format_controls()

    def _update_recording_format_controls(self) -> None:
        """Show the format row only when AAC is offered; the bitrate applies to AAC only."""
        offered = self.recording_format_selector.count() > 1
        self.recording_format_label.setVisible(offered)
        self.recording_format_selector.setVisible(offered)
        aac = offered and self.recording_format_selector.currentData() == "aac"
        for widget in (self.recording_bitrate_label, self.recording_bitrate_selector):
            widget.setEnabled(aac)
            widget.setVisible(aac)

    def _save_recording_controls(self) -> None:
        """Store the values that need conversion or validation."""
        current = self._input_language()
        self.config["input_language"] = current
        self.config["recording_palette_languages"] = self._selected_palette_languages(current)
        try:
            self.config["gain_db"] = float(self.gain_input.text() or 0)
        except (ValueError, TypeError):
            # Keep the previous gain so an invalid entry never aborts the whole save.
            logging.warning(f"Invalid gain value '{self.gain_input.text()}'; keeping previous gain {self.config.get('gain_db')}.")
            self.gain_input.setText(str(self.config.get("gain_db", 0)))
        # Until the probe populated the selectors they cannot represent a saved AAC choice.
        if self._aac_bitrates_probed:
            self.config["recording_format"] = self.recording_format_selector.currentData() or "wav"
            self.config["recording_aac_bitrate_kbps"] = self.recording_bitrate_selector.currentData() or 64
