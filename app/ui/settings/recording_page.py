"""Recording controls in the "Recording & Hotkey" card: push-to-talk, warm microphone, length, format."""
from __future__ import annotations

import logging
import threading

from PyQt6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QSpinBox, QWidget

from app.audio.aac_encoder import available_aac_bitrates
from app.core.constants import LANGUAGES
from app.core.env import is_WINDOWS
from app.ui.settings.base import SettingsWindowBase


class RecordingPage(SettingsWindowBase):
    """Recording controls, including the AAC format offered after a background encoder probe."""

    def _build_recording_controls(self) -> None:
        """Create the recording widgets inside the card that holds hotkey, language and gain."""
        layout = self.recording_group_layout
        self.push_to_talk_checkbox = QCheckBox(self)
        layout.addWidget(self.push_to_talk_checkbox)
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

        for name, code in LANGUAGES.items():
            self.language_input.addItem(name, code)
        self.language_input.setCurrentIndex(max(0, self.language_input.findData(self.config["input_language"].lower())))
        self.gain_input.setText(str(self.config["gain_db"]))

    def _add_row(self, *widgets: QWidget) -> None:
        """Append a left-aligned row of widgets to the recording card."""
        row = QHBoxLayout()
        for widget in widgets:
            row.addWidget(widget)
        row.addStretch()
        self.recording_group_layout.addLayout(row)

    def _connect_recording_controls(self) -> None:
        """Live enable/visibility updates once the saved values are loaded."""
        self.windows_keep_mic_hot_checkbox.stateChanged.connect(self._update_windows_keep_mic_hot_ui_state)
        self.recording_format_selector.currentIndexChanged.connect(self._update_recording_format_controls)
        self._update_windows_keep_mic_hot_ui_state()
        self._update_recording_format_controls()

    def _update_windows_keep_mic_hot_ui_state(self) -> None:
        """The idle timeout only applies while the warm microphone is enabled."""
        enabled = is_WINDOWS and self.windows_keep_mic_hot_checkbox.isChecked()
        self.windows_keep_mic_hot_idle_label.setEnabled(enabled)
        self.windows_keep_mic_hot_idle_input.setEnabled(enabled)

    def start_aac_bitrate_probe(self) -> None:
        """Probe the optional PyAV AAC encoder in the background; importing PyAV takes noticeable time."""
        def probe() -> None:
            try:
                bitrates = available_aac_bitrates()
            except Exception as error:
                logging.warning("AAC encoder probe failed: %s", error)
                bitrates = ()
            self.aac_bitrates_ready.emit(bitrates)

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
        self.config["input_language"] = self.language_input.currentData() or "en"
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
