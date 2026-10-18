"""Transcription page: section layout, collapsible prompt and FFmpeg sections, token counter."""
from __future__ import annotations

import logging
import threading
from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSizePolicy

from app.core.models import prompt_token_limit, transcription_supports_prompt, transcription_supports_temperature
from app.core.prompts import default_transcription_prompt, is_default_transcription_prompt
from app.core.textutil import estimate_tokens
from app.services.ffmpeg import probe_version, resolve_ffmpeg
from app.ui.settings.base import SettingsWindowBase
from app.ui.settings.collapsible import CollapsibleSection
from app.ui.settings.widgets import show_temperature_support
from app.ui.theme import set_style_state

#: The counter turns to the accent colour once the prompt uses this share of the model's budget.
_TOKEN_WARNING_SHARE = 0.85
#: Horizontal gap between the provider, model and temperature columns of an API group.
_COLUMN_GAP = 12
#: Visible text lines of the transcription prompt editor, plus its padding and border.
_PROMPT_LINES = 6
_PROMPT_FRAME_PX = 14
#: Quality fallback modes and their caption keys, in dropdown order.
_SECOND_PASS_MODES = (("off", "second_pass_mode_off"), ("doubtful", "second_pass_mode_doubtful"),
                      ("always", "second_pass_mode_always"))
#: Pause after the last keystroke before probing the FFmpeg path (a subprocess per probe).
_FFMPEG_PROBE_DEBOUNCE_MS = 400


class TranscriptionPage(SettingsWindowBase):
    """Layout of the transcription page and its live status indicators."""

    def _init_transcription_page(self) -> None:
        """Order the page: API sections, hotkeys, recording, then the collapsible FFmpeg section."""
        # Token counter renders as a compact pill badge, so it should hug its content.
        self.prompt_token_label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        # Room for the default prompt (about six lines); longer prompts scroll.
        self.prompt_input.setFixedHeight(self.prompt_input.fontMetrics().lineSpacing() * _PROMPT_LINES
                                         + _PROMPT_FRAME_PX)
        # The section header names the prompt, so its separate label is not shown.
        self.transcription_prompt_label.hide()
        self.transcription_prompt_section = CollapsibleSection(self)
        for widget in (self.prompt_input, self.prompt_token_label):
            self.transcription_layout.removeWidget(widget)
        self.restore_default_prompt_button = QPushButton(self)
        self.restore_default_prompt_button.clicked.connect(self._restore_default_transcription_prompt)
        # Token counter on the left, "restore default" on the right, below the prompt.
        prompt_footer = QHBoxLayout()
        prompt_footer.addWidget(self.prompt_token_label)
        prompt_footer.addStretch(1)
        prompt_footer.addWidget(self.restore_default_prompt_button)
        self.transcription_prompt_section.content_layout.addWidget(self.prompt_input)
        self.transcription_prompt_section.content_layout.addLayout(prompt_footer)
        self.transcription_api_layout.addWidget(self.transcription_prompt_section)
        self._build_second_pass_section()
        # Provider | model | temperature rows: columns start at the top, and the slider is as tall as
        # the dropdowns, so all three labels and controls line up.
        for row, combo, slider, columns in (
            (self.transcription_provider_model_row, self.model_dropdown, self.transcription_temp_slider,
             (self.transcription_provider_col, self.transcription_model_col, self.transcription_temp_col)),
            (self.rephrasing_provider_model_row, self.rephrasing_model_input, self.rephrasing_temp_slider,
             (self.rephrasing_provider_col, self.rephrasing_model_col, self.rephrasing_temp_col)),
        ):
            slider.setMinimumHeight(combo.sizeHint().height())
            row.setSpacing(_COLUMN_GAP)
            for index, column in enumerate(columns):
                row.setAlignment(column, Qt.AlignmentFlag.AlignTop)
                row.setStretch(index, 1)  # provider | model | temperature: a third each
        for column in (self.hotkey_v_layout, self.pr_hotkey_v_layout):
            self.hotkeys_layout.setAlignment(column, Qt.AlignmentFlag.AlignTop)
        # Language / gain row: the language name gets the room a two-digit gain does not need.
        for index, stretch in enumerate((2, 1)):
            self.controls_layout.setStretch(index, stretch)
        self._build_ffmpeg_settings_row()
        self.transcription_layout.addStretch(1)

    def _connect_transcription_page(self) -> None:
        """Live updates once the saved values are loaded."""
        self.prompt_input.textChanged.connect(self._update_prompt_token_counter)
        self.language_input.currentIndexChanged.connect(
            lambda _index: self.swap_default_transcription_prompt(self.language_input.currentData() or "",
                                                                  self.config.get("ui_language", "en")))
        self.ffmpeg_path_input.textChanged.connect(lambda _text: self._ffmpeg_status_debounce.start())
        self.second_pass_mode_selector.currentIndexChanged.connect(self._update_second_pass_controls)
        self._update_second_pass_controls()
        QTimer.singleShot(0, self._update_prompt_token_counter)
        self._refresh_ffmpeg_status()

    def _build_second_pass_section(self) -> None:
        """Collapsed "quality fallback" section below the prompt: mode, threshold and prompt of a second transcription."""
        self.second_pass_section = CollapsibleSection(self)
        content = self.second_pass_section.content_layout
        self.second_pass_mode_label = QLabel(self)
        self.second_pass_mode_selector = QComboBox(self)
        for mode, _key in _SECOND_PASS_MODES:
            self.second_pass_mode_selector.addItem("", mode)
        self.second_pass_logprob_label = QLabel(self)
        self.second_pass_logprob_input = QDoubleSpinBox(self)
        self.second_pass_logprob_input.setRange(-5.0, 0.0)
        self.second_pass_logprob_input.setSingleStep(0.05)
        self.second_pass_logprob_input.setDecimals(2)
        self.second_pass_prompt_label = QLabel(self)
        self.second_pass_prompt_input = QLineEdit(self)
        for widgets in ((self.second_pass_mode_label, self.second_pass_mode_selector),
                        (self.second_pass_logprob_label, self.second_pass_logprob_input)):
            row = QHBoxLayout()
            for widget in widgets:
                row.addWidget(widget)
            row.addStretch(1)
            content.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(self.second_pass_prompt_label)
        row.addWidget(self.second_pass_prompt_input, 1)
        content.addLayout(row)
        self.transcription_api_layout.addWidget(self.second_pass_section)

    def _retranslate_transcription_page(self) -> None:
        """Translate the quality fallback modes, whose item data stays fixed."""
        for index, (_mode, key) in enumerate(_SECOND_PASS_MODES):
            self.second_pass_mode_selector.setItemText(index, self.translator.tr(key))

    def _update_second_pass_controls(self, *_args: object) -> None:
        """Threshold and prompt only apply while a second transcription is enabled."""
        enabled = self.second_pass_mode_selector.currentData() != "off"
        for widget in (self.second_pass_logprob_label, self.second_pass_logprob_input,
                       self.second_pass_prompt_label, self.second_pass_prompt_input):
            widget.setEnabled(enabled)

    def _restore_default_transcription_prompt(self) -> None:
        """Put the default prompt for the selected input language (else the UI language) into the editor."""
        self.prompt_input.setPlainText(default_transcription_prompt(self.language_input.currentData() or "",
                                                                    self.config.get("ui_language", "en")))

    def swap_default_transcription_prompt(self, input_language: str, ui_language: str) -> None:
        """Replace an unedited default prompt with the default for the given languages; edited prompts stay."""
        current = self.prompt_input.toPlainText()
        if not is_default_transcription_prompt(current):
            return
        new_default = default_transcription_prompt(input_language, ui_language)
        if current.strip() != new_default:
            self.prompt_input.setPlainText(new_default)
            logging.info("Swapped the default transcription prompt to input=%s ui=%s.", input_language, ui_language)

    def _build_ffmpeg_settings_row(self) -> None:
        """Add the collapsible FFmpeg section (status, path field, Browse) below the recording settings.

        With FFmpeg available, video files can be picked for transcription and long recordings
        are compressed to the upload limit. An empty path means auto-detection on PATH.
        """
        self.ffmpeg_label = QLabel(self)
        self.ffmpeg_path_input = QLineEdit(self)
        self.ffmpeg_browse_button = QPushButton(self)
        self.ffmpeg_browse_button.clicked.connect(self._browse_ffmpeg_path)
        # The probe runs `ffmpeg -version` in a background thread: debounce it while typing, and
        # only the newest probe's result is shown.
        self._ffmpeg_status_debounce = QTimer(self)
        self._ffmpeg_status_debounce.setSingleShot(True)
        self._ffmpeg_status_debounce.setInterval(_FFMPEG_PROBE_DEBOUNCE_MS)
        self._ffmpeg_status_debounce.timeout.connect(self._refresh_ffmpeg_status)
        self._ffmpeg_probe_generation = 0
        self._ffmpeg_version: Optional[str] = None
        self._ffmpeg_probed = False
        self.ffmpeg_version_ready.connect(self._apply_ffmpeg_version)

        ffmpeg_row = QHBoxLayout()
        ffmpeg_row.addWidget(self.ffmpeg_path_input)
        ffmpeg_row.addWidget(self.ffmpeg_browse_button)
        self.ffmpeg_status_label = QLabel(self)
        self.ffmpeg_status_label.setObjectName("ffmpeg_status_label")
        self.ffmpeg_status_label.setWordWrap(True)

        self.ffmpeg_section = CollapsibleSection(self)
        content = self.ffmpeg_section.content_layout
        content.addWidget(self.ffmpeg_status_label)
        content.addWidget(self.ffmpeg_label)
        content.addLayout(ffmpeg_row)
        self.transcription_layout.addWidget(self.ffmpeg_section)

    def _browse_ffmpeg_path(self) -> None:
        """Pick the FFmpeg binary with a file dialog."""
        path, _ = QFileDialog.getOpenFileName(
            self, self.translator.tr("ffmpeg_browse_dialog_title"), self.ffmpeg_path_input.text().strip()
        )
        if path:
            self.ffmpeg_path_input.setText(path)

    def _refresh_ffmpeg_status(self) -> None:
        """Probe the configured path / PATH in the background; ``ffmpeg_version_ready`` brings the result."""
        self._ffmpeg_probe_generation += 1
        generation = self._ffmpeg_probe_generation
        configured = self.ffmpeg_path_input.text()

        def probe() -> None:
            version = probe_version(resolve_ffmpeg(configured))
            try:
                self.ffmpeg_version_ready.emit((generation, version))
            except RuntimeError:
                pass  # the window was deleted while the probe ran

        threading.Thread(target=probe, name="FfmpegProbe", daemon=True).start()

    def _apply_ffmpeg_version(self, result: tuple[int, Optional[str]]) -> None:
        """Keep the newest probe's version (an older probe may finish later) and show it."""
        generation, version = result
        if generation != self._ffmpeg_probe_generation:
            return
        self._ffmpeg_version = version
        self._ffmpeg_probed = True
        self._render_ffmpeg_status()

    def _render_ffmpeg_status(self) -> None:
        """Show the cached probe result in the current language; nothing until the first probe is done."""
        if not self._ffmpeg_probed:
            return
        version = self._ffmpeg_version
        if version:
            self.ffmpeg_status_label.setText(self.translator.tr("ffmpeg_status_detected", version=version))
        else:
            self.ffmpeg_status_label.setText(self.translator.tr("ffmpeg_status_not_found"))
        set_style_state(self.ffmpeg_status_label, "found", bool(version))

    def _update_prompt_token_counter(self) -> None:
        """Show the prompt's token estimate against the model's budget; disable unsupported fields."""
        model = self.model_dropdown.currentText()
        show_temperature_support(self.transcription_temp_slider, self.transcription_temp_label,
                                 transcription_supports_temperature(model), self.translator)
        self.prompt_input.setEnabled(transcription_supports_prompt(model))
        tokens = estimate_tokens(self.prompt_input.toPlainText())
        limit = prompt_token_limit(model)
        level = "normal"
        if limit:
            over = tokens > limit
            level = "over" if over else ("near" if tokens > int(limit * _TOKEN_WARNING_SHARE) else "normal")
            exceeded_text = self.translator.tr("token_exceeded_text") if over else ""
            self.prompt_token_label.setText(self.translator.tr(
                "token_counter_exceeded_label", tokens=tokens, limit=limit, exceeded_text=exceeded_text))
        else:
            self.prompt_token_label.setText(self.translator.tr("token_counter_label", tokens=tokens))
        set_style_state(self.prompt_token_label, "level", level)
