"""Modern design of the Prompts page: a card with the prompt list beside a card with the editor.

Like the other modern pages it reuses the widgets of the classic page (built from the .ui file and
driven by ``TransformationsEditor``) and only arranges them differently.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from PyQt6.QtCore import QModelIndex, QPersistentModelIndex, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from app.core.prompts import INSTRUCTION
from app.ui.durations import BALLOON_MAX_MS
from app.ui.settings.base import SettingsWindowBase
from app.ui.settings.modern_rows import rebuild_card

_LIST_WIDTH = 270
_ITEM_PADDING = 10
_ITEM_RADIUS = 8
_TEXT_MIN_HEIGHT = 160
_ENTRY_ID = Qt.ItemDataRole.UserRole

Index = QModelIndex | QPersistentModelIndex


class PromptItemDelegate(QStyledItemDelegate):
    """Two lines per prompt: its caption, and the start of its text (muted); "Auto" marks the automatic one."""

    def __init__(self, entry: Callable[[str], Optional[Dict[str, Any]]], colors: Callable[[], Dict[str, str]],
                 auto_tag: Callable[[], str], parent: Optional[QWidget] = None) -> None:
        """``entry`` looks an entry up by ID; ``colors`` returns the theme palette at paint time."""
        super().__init__(parent)
        self._entry = entry
        self._colors = colors
        self._auto_tag = auto_tag

    def sizeHint(self, option: QStyleOptionViewItem, index: Index) -> QSize:  # noqa: N802 - Qt API
        """Room for two lines of text."""
        metrics = QFontMetrics(option.font)
        return QSize(option.rect.width(), 2 * metrics.height() + 2 * _ITEM_PADDING + 4)

    def paint(self, painter: Optional[QPainter], option: QStyleOptionViewItem, index: Index) -> None:
        """Rounded selection, bold caption, muted preview."""
        if painter is None:
            return
        colors = self._colors()
        entry = self._entry(str(index.data(_ENTRY_ID))) or {}
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        box = QRectF(option.rect).adjusted(1, 2, -1, -2)
        state = option.state
        if state & QStyle.StateFlag.State_Selected:
            fill, border = QColor(colors["accent"]), QColor(colors["accent"])
            fill.setAlpha(28)
            border.setAlpha(90)
        elif state & QStyle.StateFlag.State_MouseOver:
            fill, border = QColor(colors["hover"]), QColor(colors["hover"])
        else:
            fill = border = QColor(Qt.GlobalColor.transparent)
        path = QPainterPath()
        path.addRoundedRect(box, _ITEM_RADIUS, _ITEM_RADIUS)
        painter.fillPath(path, fill)
        painter.setPen(border)
        painter.drawPath(path)

        inactive = entry.get("kind") == INSTRUCTION and not entry.get("enabled", True)
        text_box = box.adjusted(_ITEM_PADDING, _ITEM_PADDING - 2, -_ITEM_PADDING, -_ITEM_PADDING + 2)
        title_font = QFont(option.font)
        title_font.setWeight(QFont.Weight.DemiBold)
        title_metrics = QFontMetrics(title_font)
        if entry.get("auto_apply") is True:
            tag = self._auto_tag()
            tag_font = QFont(option.font)
            tag_font.setPointSizeF(max(7.0, option.font.pointSizeF() - 1.5))
            tag_font.setWeight(QFont.Weight.DemiBold)
            tag_width = QFontMetrics(tag_font).horizontalAdvance(tag) + 12
            tag_box = QRectF(text_box.right() - tag_width, text_box.top() + 1, tag_width, title_metrics.height() - 2)
            accent = QColor(colors["accent"])
            tint = QColor(accent)
            tint.setAlpha(36)
            pill = QPainterPath()
            pill.addRoundedRect(tag_box, tag_box.height() / 2, tag_box.height() / 2)
            painter.fillPath(pill, tint)
            painter.setFont(tag_font)
            painter.setPen(accent)
            painter.drawText(tag_box, Qt.AlignmentFlag.AlignCenter, tag)
            text_box.setRight(tag_box.left() - 6)
        painter.setFont(title_font)
        painter.setPen(QColor(colors["muted" if inactive else "text"]))
        title = title_metrics.elidedText(str(index.data(Qt.ItemDataRole.DisplayRole) or ""),
                                         Qt.TextElideMode.ElideRight, int(text_box.width()))
        painter.drawText(text_box, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, title)
        preview = next((line.strip() for line in str(entry.get("text", "")).splitlines() if line.strip()), "")
        painter.setFont(option.font)
        painter.setPen(QColor(colors["muted"]))
        line_box = QRectF(box.left() + _ITEM_PADDING, text_box.top() + title_metrics.height() + 2,
                          box.width() - 2 * _ITEM_PADDING, title_metrics.height())
        painter.drawText(line_box, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                         QFontMetrics(option.font).elidedText(preview, Qt.TextElideMode.ElideRight,
                                                              int(line_box.width())))
        painter.restore()


class ModernPromptsPage(SettingsWindowBase):
    """Arrange the Prompts page as list card, editor card and a card for the recording palette options."""

    def _build_modern_prompts_page(self) -> None:
        """Rebuild the Prompts page (modern design, at startup)."""
        tr = self.translator.tr
        editor = self._transformations
        prompt_list = editor.list
        prompt_list.setObjectName("promptList")
        prompt_list.setAlternatingRowColors(False)
        prompt_list.setMouseTracking(True)  # hover highlight
        prompt_list.setItemDelegate(PromptItemDelegate(editor.entry, self.theme_palette,
                                                       lambda: tr("prompts_tag_auto"), prompt_list))
        viewport = prompt_list.viewport()
        if viewport is not None:
            editor.changed.connect(viewport.update)  # previews follow the edits

        # Left: the list, the limit note and the add/remove buttons.
        self.prompts_list_group = QGroupBox(self)
        list_layout = QVBoxLayout(self.prompts_list_group)
        list_layout.setContentsMargins(10, 10, 10, 12)
        list_layout.setSpacing(8)
        self.splitter.replaceWidget(0, self.prompts_list_group)
        list_layout.addWidget(prompt_list, 1)
        # The how-to texts sit behind a "?" button, so a long prompt list keeps the card's height.
        for note in (self.transformations_tab_description_label, self.transformations_info_label):
            note.hide()
        self.prompts_help_button = QPushButton("?", self.prompts_list_group)
        self.prompts_help_button.setObjectName("liveprompt_help_button")  # the theme's round "?" badge
        self.prompts_help_button.setFixedSize(22, 22)
        self.prompts_help_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.prompts_help_button.clicked.connect(self._show_prompts_help)
        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        buttons.addWidget(self.post_rp_add_btn)
        buttons.addWidget(self.post_rp_remove_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.prompts_help_button)
        list_layout.addLayout(buttons)

        # Right: the editor of the selected entry as rows.
        self.prompts_editor_group = QGroupBox(self)
        editor_layout = QVBoxLayout(self.prompts_editor_group)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        self.splitter.replaceWidget(1, self.prompts_editor_group)
        editor_layout.addWidget(self.post_rp_editor_scroll)
        self.splitter.setHandleWidth(14)
        self.splitter.setChildrenCollapsible(False)
        self.prompts_list_group.setMinimumWidth(_LIST_WIDTH - 40)
        self.splitter.setSizes([_LIST_WIDTH, 10_000])

        row, switch = self._row, self._switch
        self.post_rp_text_edit.setMinimumHeight(_TEXT_MIN_HEIGHT)
        rebuild_card(self.post_rp_instruction_options, [
            self._row(None, self.liveprompt_help_button, self.liveprompt_enabled_checkbox,
                      checkbox=self.liveprompt_enabled_checkbox),
            row(self.liveprompt_trigger_label, self.liveprompt_trigger_words_input),
            row(self.liveprompt_trigger_scan_depth_label, self.liveprompt_trigger_scan_depth_input),
            switch(self.liveprompt_strip_trigger_checkbox),
            switch(self.rephrase_context_checkbox),
        ], margins=(0, 0, 0, 0))
        text_row = self._row(self.text_label, self.post_rp_text_edit, stacked=True)
        body = rebuild_card(self.post_rp_editor_container, [
            switch(self.post_rp_instruction_enabled_checkbox),
            row(self.caption_label, self.post_rp_caption_edit),
            switch(self.post_rp_show_during_recording_checkbox),
            switch(self.post_rp_auto_apply_checkbox),
            text_row,
            self.post_rp_instruction_options,
        ])
        body.set_stretch(text_row, 1)

        # Below: how the prompt palette behaves during a recording.
        self.prompts_palette_group = QGroupBox(self)
        QVBoxLayout(self.prompts_palette_group)
        rebuild_card(self.prompts_palette_group, [
            switch(self.recording_prompt_overlay_system_position_checkbox),
            switch(self.recording_prompt_overlay_digit_keys_checkbox),
            row(self.recording_overlay_opacity_label, self.recording_overlay_opacity_input),
        ])
        page = self.pr_layout
        page.insertWidget(page.indexOf(self.splitter) + 1, self.prompts_palette_group)
        page.setStretchFactor(self.splitter, 1)

    def _prompts_help_text(self) -> str:
        """What the Prompts page does, and the limit of templates."""
        labels = (self.transformations_tab_description_label, self.transformations_info_label)
        return "\n\n".join(label.text() for label in labels if label.text())

    def _show_prompts_help(self) -> None:
        """Explain the Prompts page in a long balloon, like the LivePrompt help."""
        self.ctx.notifier.show(self._prompts_help_text(), BALLOON_MAX_MS)

    def _retranslate_modern_prompts_page(self) -> None:
        """Card titles of the modern Prompts page."""
        tr = self.translator.tr
        self.prompts_list_group.setTitle(tr("prompts_list_title"))
        self.prompts_editor_group.setTitle(tr("prompts_editor_title"))
        self.prompts_palette_group.setTitle(tr("prompts_palette_group_title"))
        self.prompts_help_button.setToolTip(self._prompts_help_text())
