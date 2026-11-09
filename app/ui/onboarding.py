"""The first-start guide: app language, spoken language, transcription and rephrasing, one question per page."""
from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QColor, QMouseEvent
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.context import AppContext
from app.core.api_keys import PROVIDER_NAMES, key_format_warning
from app.core.constants import APP_DATA_DIR
from app.core.hotkeys import pretty_hotkey
from app.core.i18n import UI_LANGUAGES
from app.core.onboarding import ApiAccess, OnboardingChoices, apply_languages, apply_onboarding
from app.ui import theme
from app.ui.language_names import DETECT_LANGUAGE, input_language_items
from app.ui.tray_icons import app_icon

#: The visible card; the frameless window around it leaves room for the drop shadow.
_CARD_WIDTH = 460
_CARD_SIDE_MARGIN = 32
_SHADOW_MARGIN = 20
_SHADOW_BLUR = 36
_SHADOW_COLOR = QColor(0, 0, 0, 110)
_LOGO_SIZE = 56
_STEP_DOT_SIZE = 8
#: Providers with an official endpoint; a custom server is set up in the settings.
_GUIDE_PROVIDERS = tuple(provider for provider in PROVIDER_NAMES if provider != "custom")


def _guide_stylesheet(dark: bool) -> str:
    """The guide's own look on top of the shared stylesheet: a floating card with selectable answers."""
    colors = theme.palette(dark)
    return f"""
QDialog#guideWindow {{ background: transparent; }}
QFrame#guideCard {{ background: {colors['bg']}; border: 1px solid {colors['border']}; border-radius: 18px; }}
QLabel#guideWelcome {{ color: {colors['muted']}; font-size: 13px; }}
QLabel#guideTitle {{ color: {colors['text']}; font-size: 18px; font-weight: 600; }}
QPushButton#guideSkip {{ background: transparent; border: none; color: {colors['muted']}; }}
QPushButton#guideSkip:hover {{ color: {colors['text']}; }}
QRadioButton {{
    padding: 11px 12px; border: 1px solid {colors['border']}; border-radius: 10px;
    background: {colors['panel']}; color: {colors['text']};
}}
QRadioButton:hover {{ border-color: {colors['muted']}; }}
QRadioButton:checked {{ border-color: {colors['accent']}; }}
QRadioButton::indicator {{
    width: 12px; height: 12px; border-radius: 7px; border: 1px solid {colors['muted']};
}}
QRadioButton::indicator:checked {{ background: {colors['accent']}; border-color: {colors['accent']}; }}
"""


class _AccessForm(QWidget):
    """Provider and API key, with the hint for a key that does not look like the provider's."""

    def __init__(self, ctx: AppContext, parent: QWidget) -> None:
        """Build the form; the texts come from ``retranslate``."""
        super().__init__(parent)
        self._tr = ctx.tr
        self.provider = QComboBox(self)
        for provider in _GUIDE_PROVIDERS:
            self.provider.addItem(PROVIDER_NAMES[provider], provider)
        self.key = QLineEdit(self)
        self.key.setEchoMode(QLineEdit.EchoMode.Password)
        self._hint = QLabel(self)
        self._hint.setWordWrap(True)
        self._provider_label = QLabel(self)
        self._key_label = QLabel(self)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow(self._provider_label, self.provider)
        form.addRow(self._key_label, self.key)
        form.addRow(self._hint)
        self.provider.currentIndexChanged.connect(self._update_hint)
        self.key.textChanged.connect(self._update_hint)

    def access(self) -> ApiAccess:
        """The current answers."""
        return ApiAccess(self.provider.currentData(), self.key.text().strip())

    def retranslate(self) -> None:
        """Apply the guide's current language."""
        self._provider_label.setText(self._tr("onboarding_provider"))
        self._key_label.setText(self._tr("onboarding_key"))
        self._update_hint()

    def _update_hint(self, *_args: object) -> None:
        """Without a key: it can follow later. With an unusual one: say so, but never refuse it."""
        access = self.access()
        if not access.key:
            self._hint.setText(self._tr("onboarding_key_later"))
            return
        warning = key_format_warning(access.provider, access.key)
        self._hint.setText(self._tr(warning) if warning else "")


class OnboardingDialog(QDialog):
    """Asks the four questions; ``choices`` reads the answers, the caller stores them."""

    def __init__(self, ctx: AppContext, *, local_supported: bool) -> None:
        """Build the pages with the configured languages preselected."""
        super().__init__()
        self._ctx = ctx
        self._local_supported = local_supported
        self.finished_guide = False
        self._radio_texts: List[Tuple[QRadioButton, str]] = []
        self._dark = theme.resolve_dark(ctx.config.get("color_theme", "system"), QApplication.instance())
        self._welcome = QLabel(self)
        self._welcome.setObjectName("guideWelcome")
        self._title = QLabel(self)
        self._title.setObjectName("guideTitle")
        self._title.setWordWrap(True)
        self._pages = QStackedWidget(self)
        self._pages.addWidget(self._build_ui_language_page())
        self._pages.addWidget(self._build_input_language_page())
        self._pages.addWidget(self._build_transcription_page())
        self._pages.addWidget(self._build_rephrasing_page())
        self._skip = QPushButton(self)
        self._skip.setObjectName("guideSkip")
        self._skip.setCursor(Qt.CursorShape.PointingHandCursor)
        self._back = QPushButton(self)
        self._next = QPushButton(self)
        self._next.setObjectName("save_button")  # the stylesheet's primary button
        self._next.setDefault(True)
        self._skip.clicked.connect(self.reject)
        self._back.clicked.connect(lambda: self._go_to(self._pages.currentIndex() - 1))
        self._next.clicked.connect(self._advance)
        buttons = QHBoxLayout()
        buttons.addWidget(self._skip)
        buttons.addStretch()
        buttons.addWidget(self._back)
        buttons.addWidget(self._next)
        self._build_window(buttons)
        self._ui_language.currentIndexChanged.connect(self._on_ui_language_changed)
        self._go_to(0)

    def _build_window(self, buttons: QHBoxLayout) -> None:
        """A small frameless window of its own: a shadowed card with the logo centered on top."""
        self.setObjectName("guideWindow")
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setWindowIcon(app_icon())
        self.setStyleSheet(theme.build_stylesheet(self._dark) + self._icon_stylesheet()
                           + _guide_stylesheet(self._dark))
        theme.round_combo_popups(self)
        self.setFixedWidth(_CARD_WIDTH + 2 * _SHADOW_MARGIN)

        card = QFrame(self)
        card.setObjectName("guideCard")
        self._card = card
        shadow = QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(_SHADOW_BLUR)
        shadow.setOffset(0, 6)
        shadow.setColor(_SHADOW_COLOR)
        card.setGraphicsEffect(shadow)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(_SHADOW_MARGIN, _SHADOW_MARGIN, _SHADOW_MARGIN, _SHADOW_MARGIN)
        outer.addWidget(card)

        logo = QLabel(card)
        logo.setPixmap(app_icon().pixmap(QSize(_LOGO_SIZE, _LOGO_SIZE), self.devicePixelRatioF()))
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(_CARD_SIDE_MARGIN, 24, _CARD_SIDE_MARGIN, 20)
        layout.setSpacing(8)
        layout.addWidget(logo, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self._welcome, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addLayout(self._build_step_dots())
        layout.addSpacing(4)
        layout.addWidget(self._title)
        layout.addSpacing(4)
        layout.addWidget(self._pages)
        layout.addSpacing(6)
        layout.addLayout(buttons)

    def _icon_stylesheet(self) -> str:
        """The dropdown arrows; they are cosmetic, so a folder that cannot be written must not stop the guide."""
        try:
            return theme.write_icon_qss(self._dark, os.path.join(APP_DATA_DIR, "theme_icons"))
        except OSError as error:
            logging.warning("Guide icons not written: %s", error)
            return ""

    def _fit_to_page(self) -> None:
        """The window is only as tall as the current question needs."""
        page = self._pages.currentWidget()
        page_layout = page.layout() if page is not None else None
        card_layout = self._card.layout()
        if page_layout is None or card_layout is None:
            return
        content_width = _CARD_WIDTH - 2 * _CARD_SIDE_MARGIN
        page_layout.invalidate()
        # Only a page with wrapping text has a height-for-width; the others report nothing useful there.
        self._pages.setFixedHeight(page_layout.totalHeightForWidth(content_width) if page_layout.hasHeightForWidth()
                                   else page_layout.totalSizeHint().height())
        card_layout.invalidate()
        self.setFixedHeight(card_layout.totalHeightForWidth(_CARD_WIDTH) + 2 * _SHADOW_MARGIN)

    def _build_step_dots(self) -> QHBoxLayout:
        """One small dot per question; ``_show_progress`` colours those reached so far."""
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addStretch()
        self._step_dots: List[QFrame] = []
        for _page in range(self._pages.count()):
            dot = QFrame(self)
            dot.setFixedSize(_STEP_DOT_SIZE, _STEP_DOT_SIZE)
            self._step_dots.append(dot)
            row.addWidget(dot)
        row.addStretch()
        return row

    def _show_progress(self, index: int) -> None:
        """Colour the dots up to the current question."""
        colors = theme.palette(self._dark)
        for position, dot in enumerate(self._step_dots):
            color = colors["accent"] if position <= index else colors["border"]
            dot.setStyleSheet(f"background: {color}; border: none; border-radius: {_STEP_DOT_SIZE // 2}px;")

    def mousePressEvent(self, event: Optional[QMouseEvent]) -> None:  # noqa: N802 - Qt API
        """A frameless window moves by dragging any free spot of its card."""
        handle = self.windowHandle()
        if event is not None and event.button() == Qt.MouseButton.LeftButton and handle is not None:
            handle.startSystemMove()

    # --- Pages ------------------------------------------------------------------------------
    def _build_ui_language_page(self) -> QWidget:
        page = QWidget(self)
        self._ui_language = QComboBox(page)
        for code, name in UI_LANGUAGES.items():
            self._ui_language.addItem(name, code)
        self._ui_language.setCurrentIndex(max(0, self._ui_language.findData(self._ctx.config.get("ui_language"))))
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._ui_language)
        return page

    def _build_input_language_page(self) -> QWidget:
        page = QWidget(self)
        self._input_language = QComboBox(page)
        for name, code in input_language_items():
            self._input_language.addItem(name, code)
        self._input_language.setCurrentIndex(max(0, self._input_language.findData(self._ctx.config.get("input_language"))))
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._input_language)
        return page

    def _radio(self, page: QWidget, group: QButtonGroup, text_key: str) -> QRadioButton:
        """A radio button whose caption follows the guide's language."""
        button = QRadioButton(page)
        group.addButton(button)
        self._radio_texts.append((button, text_key))
        return button

    def _build_transcription_page(self) -> QWidget:
        page = QWidget(self)
        group = QButtonGroup(page)
        self._use_api = self._radio(page, group, "onboarding_use_api")
        self._use_local = self._radio(page, group, "onboarding_use_local")
        self._use_local.setVisible(self._local_supported)
        self._use_api.setChecked(True)
        self._transcription_access = _AccessForm(self._ctx, page)
        self._local_hint = QLabel(page)
        self._local_hint.setWordWrap(True)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        for widget in (self._use_api, self._use_local, self._transcription_access, self._local_hint):
            layout.addWidget(widget)
        self._use_api.toggled.connect(self._update_answer_controls)
        return page

    def _build_rephrasing_page(self) -> QWidget:
        page = QWidget(self)
        group = QButtonGroup(page)
        self._rephrasing_off = self._radio(page, group, "onboarding_rephrasing_off")
        self._rephrasing_on = self._radio(page, group, "onboarding_rephrasing_on")
        self._rephrasing_off.setChecked(True)
        self._rephrasing_access = _AccessForm(self._ctx, page)
        self._same_key_hint = QLabel(page)
        self._same_key_hint.setWordWrap(True)
        self._hotkey_hint = QLabel(page)
        self._hotkey_hint.setWordWrap(True)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        for widget in (self._rephrasing_off, self._rephrasing_on, self._rephrasing_access, self._same_key_hint,
                       self._hotkey_hint):
            layout.addWidget(widget)
        self._rephrasing_on.toggled.connect(self._update_answer_controls)
        return page

    # --- Flow -------------------------------------------------------------------------------
    def _on_ui_language_changed(self, _index: int) -> None:
        """The guide switches at once; the spoken language follows until it is set on its own page."""
        code = self._ui_language.currentData()
        self._ctx.translator.set_language(code)
        self._input_language.setCurrentIndex(max(0, self._input_language.findData(code)))
        self._retranslate()

    def _advance(self) -> None:
        """Go on, or finish on the last page."""
        if self._pages.currentIndex() == self._pages.count() - 1:
            self.finished_guide = True
            self.accept()
            return
        self._go_to(self._pages.currentIndex() + 1)

    def _go_to(self, index: int) -> None:
        self._pages.setCurrentIndex(index)
        self._show_progress(index)
        self._retranslate()

    def _shared_key_access(self) -> Optional[ApiAccess]:
        """The transcription provider and key, when rephrasing can simply use them too."""
        if not self._use_api.isChecked():
            return None
        access = self._transcription_access.access()
        return access if access.key else None

    def _update_answer_controls(self, *_args: object) -> None:
        """Only the controls the current answers need are shown."""
        self._transcription_access.setVisible(self._use_api.isChecked())
        self._local_hint.setVisible(not self._use_api.isChecked())
        shared = self._shared_key_access()
        wanted = self._rephrasing_on.isChecked()
        self._rephrasing_access.setVisible(wanted and shared is None)
        self._same_key_hint.setVisible(wanted and shared is not None)
        if shared is not None:
            self._same_key_hint.setText(self._ctx.tr("onboarding_rephrasing_same_key",
                                                     provider=PROVIDER_NAMES[shared.provider]))
        self._fit_to_page()

    def _retranslate(self) -> None:
        """Apply the current language to every text of the guide."""
        tr = self._ctx.tr
        index = self._pages.currentIndex()
        last = index == self._pages.count() - 1
        titles = ("onboarding_ui_title", "onboarding_input_title", "onboarding_transcription_title",
                  "onboarding_rephrasing_title")
        self.setWindowTitle(tr("onboarding_window_title"))
        self._welcome.setText(tr("onboarding_window_title"))
        self._title.setText(tr(titles[index]))
        self._skip.setText(tr("onboarding_skip"))
        self._back.setText(tr("onboarding_back"))
        self._back.setVisible(index > 0)
        self._next.setText(tr("onboarding_finish" if last else "onboarding_next"))
        for button, text_key in self._radio_texts:
            button.setText(tr(text_key))
        self._input_language.setItemText(self._input_language.findData(DETECT_LANGUAGE), tr("input_language_detect"))
        self._local_hint.setText(tr("onboarding_local_hint"))
        self._hotkey_hint.setText(tr("onboarding_hotkey_hint",
                                     hotkey=pretty_hotkey(str(self._ctx.config.get("hotkey", "")))))
        self._transcription_access.retranslate()
        self._rephrasing_access.retranslate()
        self._update_answer_controls()

    # --- Answers ----------------------------------------------------------------------------
    def languages(self) -> Tuple[str, str]:
        """(app language, spoken language)."""
        return self._ui_language.currentData(), self._input_language.currentData()

    def choices(self) -> OnboardingChoices:
        """All answers, for a guide that was finished."""
        ui_language, input_language = self.languages()
        rephrasing: Optional[ApiAccess] = None
        if self._rephrasing_on.isChecked():
            rephrasing = self._shared_key_access() or self._rephrasing_access.access()
        return OnboardingChoices(ui_language, input_language,
                                 self._transcription_access.access() if self._use_api.isChecked() else None,
                                 rephrasing)


def run_onboarding(ctx: AppContext, *, local_supported: bool) -> None:
    """Show the guide once and store the answers; skipping keeps only the languages chosen so far."""
    dialog = OnboardingDialog(ctx, local_supported=local_supported)
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    dialog.exec()
    if dialog.finished_guide:
        apply_onboarding(ctx.config, dialog.choices())
    else:
        apply_languages(ctx.config, *dialog.languages())
    ctx.config["onboarding_done"] = True
    ctx.save_config()
    logging.info("onboarding finished=%s ui=%s input=%s rephrasing=%s local_setup=%s", dialog.finished_guide,
                 ctx.config.get("ui_language"), ctx.config.get("input_language"),
                 ctx.config.get("rephrasing_enabled"), ctx.config.get("local_engine_setup_pending"))
    dialog.deleteLater()
