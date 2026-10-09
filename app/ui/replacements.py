"""Settings page for transcript correction rules shared with Turbo-Type."""
from __future__ import annotations

from typing import Optional

from PyQt6.QtGui import QColor, QSyntaxHighlighter, QTextDocument
from PyQt6.QtWidgets import QCheckBox, QLabel, QPlainTextEdit, QVBoxLayout, QWidget

from app.core.i18n import TranslationManager


class ReplacementsHighlighter(QSyntaxHighlighter):
    """Color source, replacement and fixed-spelling fields without changing rule text."""

    def __init__(self, document: QTextDocument) -> None:
        """Attach Qt's incremental highlighting to the existing plain-text document."""
        super().__init__(document)
        self.set_theme(False)

    def set_theme(self, dark: bool) -> None:
        """Keep the three field colors readable on the current background."""
        self._colors = [QColor(color) for color in
                        (("#7bbcff", "#76d4a2", "#f2ba73") if dark else ("#2466ab", "#197449", "#985c13"))]
        self.rehighlight()

    def highlightBlock(self, text: Optional[str]) -> None:  # noqa: N802 - Qt API
        """Highlight each semicolon-delimited field; separators retain the normal color."""
        if not text:
            return
        offset = 0
        for field, color in zip(text.split(";"), self._colors):
            # Qt positions count UTF-16 units, including two units for an emoji.
            length = len(field.encode("utf-16-le", "surrogatepass")) // 2
            self.setFormat(offset, length, color)
            offset += length + 1


class ReplacementsTab(QWidget):
    """Edit the complete rule list; the main Save action validates and applies it."""

    def __init__(self, raw: str, enabled: bool, translator: TranslationManager, parent: Optional[QWidget] = None) -> None:
        """Build a compact page whose rule editor shrinks with the window."""
        super().__init__(parent)
        self.translator = translator
        layout = QVBoxLayout(self)
        self.enabled = QCheckBox(self)
        self.enabled.setChecked(enabled)
        layout.addWidget(self.enabled)
        self.description = QLabel(self)
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        self.editor = QPlainTextEdit(self)
        document = self.editor.document()
        assert document is not None
        self.highlighter = ReplacementsHighlighter(document)
        self.editor.setPlainText(raw)
        layout.addWidget(self.editor, 1)
        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        """Translate labels while preserving the user's rule text."""
        tr = self.translator.tr
        self.enabled.setText(tr("replacements_enabled"))
        self.description.setText(tr("replacements_description"))
        self.editor.setAccessibleName(tr("tab_replacements"))
        self.editor.setPlaceholderText("Croc, Krog, Krok ; Groq ; 1\nchat gpt ; ChatGPT ; 1")
