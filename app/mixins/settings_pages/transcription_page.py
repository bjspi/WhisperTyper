"""Transcription page: layout, FFmpeg path row and the prompt token counter."""
from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSizePolicy

from app.core.models import prompt_token_limit, transcription_supports_prompt, transcription_supports_temperature
from app.core.textutil import estimate_tokens
from app.services.ffmpeg import probe_version, resolve_ffmpeg
from app.ui.theme import set_style_state

#: The counter turns to the accent colour once the prompt uses this share of the model's budget.
_TOKEN_WARNING_SHARE = 0.85
#: Pause after the last keystroke before probing the FFmpeg path (a blocking subprocess).
_FFMPEG_PROBE_DEBOUNCE_MS = 400


class TranscriptionPageMixin:
    """Layout of the transcription page and its live status indicators."""

    def _init_transcription_page(self) -> None:
        """Size the page's rows and add the FFmpeg row above the prompt."""
        # Token counter renders as a compact pill badge, so it should hug its content.
        self.prompt_token_label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        # Keep the controls at their preferred height; the prompt takes the resizing space.
        self.prompt_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self.prompt_input.setMinimumHeight(80)
        self.transcription_layout.setStretch(self.transcription_layout.indexOf(self.prompt_input), 1)
        # Model and temperature share one row 50:50, top-aligned so both captions line up.
        self.model_temp_row.setStretch(0, 1)
        self.model_temp_row.setStretch(1, 1)
        self.model_temp_row.setSpacing(8)
        for column in (self.model_col, self.temp_col):
            self.model_temp_row.setAlignment(column, Qt.AlignmentFlag.AlignTop)
        # Hotkey / language / gain row: long hotkey combos get the room a two-digit gain does not need.
        for index, stretch in enumerate((3, 2, 1)):
            self.controls_layout.setStretch(index, stretch)
        self._build_ffmpeg_settings_row()

    def _connect_transcription_page(self) -> None:
        """Live updates once the saved values are loaded."""
        self.prompt_input.textChanged.connect(self._update_prompt_token_counter)
        self.ffmpeg_path_input.textChanged.connect(lambda _text: self._ffmpeg_status_debounce.start())
        QTimer.singleShot(0, self._update_prompt_token_counter)
        self._refresh_ffmpeg_status()

    def _build_ffmpeg_settings_row(self) -> None:
        """Add the FFmpeg path row (label, path field, Browse, status) above the transcription prompt.

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

        insert_at = self.transcription_layout.indexOf(self.transcription_prompt_label)
        if insert_at < 0:
            insert_at = self.transcription_layout.count()
        self.transcription_layout.insertWidget(insert_at, self.ffmpeg_status_label)
        self.transcription_layout.insertLayout(insert_at, ffmpeg_row)
        self.transcription_layout.insertWidget(insert_at, self.ffmpeg_label)

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
        temperature_supported = transcription_supports_temperature(model)
        self.transcription_temp_slider.setEnabled(temperature_supported)
        self.transcription_temp_label.setEnabled(temperature_supported)
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
