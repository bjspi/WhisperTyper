"""The question before picked files are transcribed: in which language, and which files get split.

The spoken language starts at the one of the settings and applies to this transcription only
("Detect language" lets the model find it). Files too long for one upload even at the minimum
bitrate are listed with the number of parts they will be cut into, so a long wait or a slightly
different result at the cuts comes as no surprise.
"""
from __future__ import annotations

import html
import logging
import os
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget

from app.core.constants import APP_DATA_DIR
from app.services.transcription_requests import UploadLimits
from app.ui import message_dialog, theme
from app.ui.language_names import DETECT_LANGUAGE, input_language_items

#: Looks up a translated text: ``tr(key, **values)``.
Translate = Callable[..., str]
_BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class SplitFile:
    """A picked file that is transcribed in ``parts`` uploads."""

    path: str
    parts: int


class FileTranscriptionDialog(message_dialog.MessageDialog):
    """Pick the spoken language for the picked files; ``language`` holds the choice after ``exec()``."""

    def __init__(self, parent: Optional[QWidget], tr: Translate, paths: Sequence[str], language: str,
                 splits: Sequence[SplitFile], limits: UploadLimits, dark: bool) -> None:
        """Preselect ``language`` (the settings' input language) and list the files in ``splits``."""
        self._start = tr("file_transcribe_start_button")
        text = (tr("file_transcribe_text_single", filename=html.escape(os.path.basename(paths[0]))) if len(paths) == 1
                else tr("file_transcribe_text_batch", count=len(paths)))
        if splits:
            text += "<br><br>" + tr("file_transcribe_split_text", limit=round(limits.max_bytes / _BYTES_PER_MB),
                                    bitrate=limits.min_bitrate_kbps)
        facts = [(os.path.basename(split.path), tr("file_transcribe_split_parts", parts=split.parts))
                 for split in splits]
        super().__init__(parent, tr("file_transcribe_title"), text,
                         [(tr("dialog_cancel_button"), message_dialog.SECONDARY), (self._start, message_dialog.PRIMARY)],
                         message_dialog.WARNING if splits else message_dialog.QUESTION,
                         window_title=tr("file_transcribe_window_title"), facts=facts)
        self._languages = QComboBox(self)
        for caption, code in input_language_items():
            self._languages.addItem(tr("input_language_detect") if code == DETECT_LANGUAGE else caption, code)
        self._languages.setCurrentIndex(max(0, self._languages.findData(language.lower())))
        self._languages.setStyleSheet(theme.build_stylesheet(dark) + _arrow_stylesheet(dark))
        theme.round_combo_popups(self)
        label = QLabel(tr("file_transcribe_language_label"), self)
        label.setObjectName("messageText")
        row = QHBoxLayout()
        row.setSpacing(12)
        row.addWidget(label)
        row.addWidget(self._languages, 1)
        self.body_layout.addLayout(row)
        self._fit(wide=False)

    @property
    def confirmed(self) -> bool:
        """Whether the transcription was started (not cancelled or closed)."""
        return self.clicked == self._start

    @property
    def language(self) -> str:
        """The chosen language code; "" lets the model detect it."""
        return str(self._languages.currentData())


def _arrow_stylesheet(dark: bool) -> str:
    """The dropdown arrow; it is cosmetic, so a folder that cannot be written must not stop the dialog."""
    try:
        return theme.write_icon_qss(dark, os.path.join(APP_DATA_DIR, "theme_icons"))
    except OSError as error:
        logging.warning("Dialog icons not written: %s", error)
        return ""


def ask_language(parent: Optional[QWidget], tr: Translate, paths: Sequence[str], language: str,
                 splits: Sequence[SplitFile], limits: UploadLimits, dark: bool) -> Optional[str]:
    """The language to transcribe ``paths`` in ("" detects it), or None when the user cancelled."""
    dialog = FileTranscriptionDialog(parent, tr, paths, language, splits, limits, dark)
    dialog.exec()
    chosen = dialog.language if dialog.confirmed else None
    dialog.deleteLater()
    return chosen
