"""Transcription page: section layout, collapsible prompt and FFmpeg sections, token counter."""
from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSizePolicy

from app.core.models import prompt_token_limit, transcription_supports_prompt, transcription_supports_temperature
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
#: Pause after the last keystroke before probing the FFmpeg path (a blocking subprocess).
_FFMPEG_PROBE_DEBOUNCE_MS = 400


class TranscriptionPage(SettingsWindowBase):
    """Layout of the transcription page and its live status indicators."""

    def _init_transcription_page(self) -> None:
        """Order the page: API sections, hotkeys, recording, then the collapsible FFmpeg section."""
        # Token counter renders as a compact pill badge, so it should hug its content.
        self.prompt_token_label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.prompt_input.setMinimumHeight(110)
        # The section header names the prompt, so its separate label is not shown.
        self.transcription_prompt_label.hide()
        self.transcription_prompt_section = CollapsibleSection(self)
        for widget in (self.prompt_input, self.prompt_token_label):
            self.transcription_layout.removeWidget(widget)
            self.transcription_prompt_section.content_layout.addWidget(widget)
        self.transcription_api_layout.addWidget(self.transcription_prompt_section)
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
        self.ffmpeg_path_input.textChanged.connect(lambda _text: self._ffmpeg_status_debounce.start())
        QTimer.singleShot(0, self._update_prompt_token_counter)
        self._refresh_ffmpeg_status()

    def _build_ffmpeg_settings_row(self) -> None:
        """Add the collapsible FFmpeg section (status, path field, Browse) below the recording settings.

        With FFmpeg available, video files can be picked for transcription and long recordings
        are compressed to the upload limit. An empty path means auto-detection on PATH.
        """
        self.ffmpeg_label = QLabel(self)
        self.ffmpeg_path_input = QLineEdit(self)
        self.ffmpeg_browse_button = QPushButton(self)
        self.ffmpeg_browse_button.clicked.connect(self._browse_ffmpeg_path)
        # The probe runs `ffmpeg -version` as a blocking subprocess: debounce it while typing.
        self._ffmpeg_status_debounce = QTimer(self)
        self._ffmpeg_status_debounce.setSingleShot(True)
        self._ffmpeg_status_debounce.setInterval(_FFMPEG_PROBE_DEBOUNCE_MS)
        self._ffmpeg_status_debounce.timeout.connect(self._refresh_ffmpeg_status)

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
        """Probe the configured path / PATH and show whether FFmpeg is available."""
        version = probe_version(resolve_ffmpeg(self.ffmpeg_path_input.text()))
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
