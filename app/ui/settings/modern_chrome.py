"""Modern design of the settings window: sidebar navigation, page header, footer bar and smaller pages.

The design is chosen at startup (switching it takes a restart). The classic design keeps the tabbed
window of the .ui file; the modern one wraps the same pages in the chrome built here.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app import __version__
from app.core.constants import WINDOW_MIN_WIDTH
from app.ui.line_icons import RoundedMenu, line_icon
from app.ui.settings.base import SettingsWindowBase
from app.ui.settings.modern_rows import CardBody, SettingRow, clear_layout, rebuild_card
from app.ui.tray_icons import app_icon

_SIDEBAR_WIDTH = 210
_BRAND_ICON_SIZE = 34
_NAV_ICON_SIZE = 18
#: Margins around the page header and the pages in the modern design.
_PAGE_MARGINS = (20, 18, 20, 6)
#: The modern window is this much wider, so its pages get the classic window's width.
_EXTRA_WIDTH = _SIDEBAR_WIDTH + _PAGE_MARGINS[0] + _PAGE_MARGINS[2]
_FOOTER_MARGINS = (24, 12, 24, 12)
#: Space left above and below the window when it grows to fit its pages (title bar, taskbar).
_SCREEN_HEADROOM = 60
#: Inner margin of the pages (from the .ui file); the page header uses it too.
_PAGE_INSET = 8


class ModernChrome(SettingsWindowBase):
    """Build the modern chrome around the pages and switch between it and the classic tabs."""

    _brand_header: QWidget
    _brand_row: QHBoxLayout
    _brand_hotkey_badge: QLabel
    brand_play_button: QPushButton
    _modern: bool = False
    _design_arranged: bool = False

    def _install_modern_chrome(self) -> None:
        """Wrap the window's content in a column beside a sidebar (hidden in the classic design)."""
        column = QWidget(self)
        self._column_layout = QVBoxLayout(column)
        self._column_layout.setSpacing(self.main_layout.spacing())
        # The page header and the tabs share an area with page margins; menu bar and footer span the width.
        self._page_area = QVBoxLayout()
        self._page_area.setSpacing(self.main_layout.spacing())
        self._page_header = self._build_page_header()
        self._page_area.addWidget(self._page_header)
        self.footer = QFrame(self)
        self.footer.setObjectName("footer")
        while self.main_layout.count():
            item = self.main_layout.takeAt(0)
            if item is None:
                continue
            widget, layout = item.widget(), item.layout()
            if widget is self.tabs:
                self._page_area.addWidget(widget, 1)
                self._column_layout.addLayout(self._page_area, 1)
            elif widget is not None:
                self._column_layout.addWidget(widget)
            elif layout is self.footer_row:
                layout.setParent(None)
                self.footer.setLayout(layout)
                self._column_layout.addWidget(self.footer)
            elif isinstance(layout, QLayout):
                layout.setParent(None)
                self._column_layout.addLayout(layout)
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_sidebar())
        body.addWidget(column, 1)
        self.main_layout.addLayout(body)
        self.tabs.currentChanged.connect(self._on_page_changed)

    def _build_page_header(self) -> QWidget:
        """Title and one-line explanation of the current page."""
        header = QWidget(self)
        header.setObjectName("pageHeader")
        layout = QVBoxLayout(header)
        # Indented like the pages' cards, so the title lines up with them.
        layout.setContentsMargins(_PAGE_INSET, 0, _PAGE_INSET, 4)
        layout.setSpacing(2)
        self._page_title = QLabel(header)
        self._page_title.setObjectName("pageTitle")
        self._page_subtitle = QLabel(header)
        self._page_subtitle.setObjectName("pageSubtitle")
        self._page_subtitle.setWordWrap(True)
        layout.addWidget(self._page_title)
        layout.addWidget(self._page_subtitle)
        return header

    def _build_sidebar(self) -> QFrame:
        """Brand, one navigation entry per page, and the hotkey, playback and menu below."""
        self._sidebar = QFrame(self)
        self._sidebar.setObjectName("sidebar")
        self._sidebar.setFixedWidth(_SIDEBAR_WIDTH)
        layout = QVBoxLayout(self._sidebar)
        layout.setContentsMargins(12, 18, 12, 14)
        layout.setSpacing(6)

        brand = QHBoxLayout()
        brand.setContentsMargins(6, 0, 0, 0)
        brand.setSpacing(10)
        logo = QLabel(self._sidebar)
        icon = app_icon()
        if not icon.isNull():
            logo.setPixmap(icon.pixmap(_BRAND_ICON_SIZE, _BRAND_ICON_SIZE))
        texts = QVBoxLayout()
        texts.setSpacing(0)
        name = QLabel("WhisperTyper", self._sidebar)
        name.setObjectName("brandName")
        self._brand_version = QLabel(self._sidebar)
        self._brand_version.setObjectName("brandVersion")
        texts.addWidget(name)
        texts.addWidget(self._brand_version)
        brand.addWidget(logo)
        brand.addLayout(texts, 1)
        layout.addLayout(brand)
        layout.addSpacing(12)

        self._nav = QListWidget(self._sidebar)
        self._nav.setObjectName("nav")
        self._nav.setFrameShape(QFrame.Shape.NoFrame)
        self._nav.setIconSize(QSize(_NAV_ICON_SIZE, _NAV_ICON_SIZE))
        self._nav.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._nav.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._nav.currentRowChanged.connect(self._on_nav_row_changed)
        layout.addWidget(self._nav, 1)

        # The hotkey badge and the play button move here while the modern design is active.
        self._sidebar_badge = QHBoxLayout()
        self._sidebar_badge.addStretch(1)
        layout.addLayout(self._sidebar_badge)
        self._more_menu = RoundedMenu(self)
        for action in (self.open_config_action, self.open_log_file_action, None, self.about_action,
                       self.github_action, None, self.exit_action):
            if action is None:
                self._more_menu.addSeparator()
            else:
                self._more_menu.addAction(action)
        # A tool button honours the stylesheet padding, so its icon lines up with the navigation icons.
        self._more_button = QToolButton(self._sidebar)
        self._more_button.setObjectName("sidebarButton")
        self._more_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._more_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._more_button.setMenu(self._more_menu)
        self._more_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._more_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._sidebar_tools = QHBoxLayout()
        self._sidebar_tools.setSpacing(4)
        self._sidebar_tools.addWidget(self._more_button, 1)
        layout.addLayout(self._sidebar_tools)
        return self._sidebar

    # --- Navigation ---------------------------------------------------------------------
    def _on_nav_row_changed(self, row: int) -> None:
        """Show the page of the clicked sidebar entry."""
        if row >= 0:
            self.tabs.setCurrentIndex(row)

    def _on_page_changed(self, index: int) -> None:
        """Keep the sidebar selection and the page header in step with the current page."""
        if self._nav.count() and self._nav.currentRow() != index:
            self._nav.setCurrentRow(index)
        self._page_title.setText(self.tabs.tabText(index))
        self._page_subtitle.setText(self.tabs.tabToolTip(index))

    def _retranslate_modern_chrome(self) -> None:
        """Sidebar entries and page header follow the tab captions; the subtitles are the tab tooltips."""
        tr = self.translator.tr
        for page, _icon, title_key, subtitle_key in self._nav_pages:
            index = self.tabs.indexOf(page)
            self.tabs.setTabText(index, tr(title_key))
            self.tabs.setTabToolTip(index, tr(subtitle_key))
            item = self._nav.item(index)
            if item is not None:
                item.setText(tr(title_key))
        self._on_page_changed(self.tabs.currentIndex())
        self._brand_version.setText(tr("brand_version", version=__version__))
        self._more_button.setText(tr("sidebar_more_button"))
        if self._rows:
            self.basics_group.setTitle(tr("basics_group_title").replace("&", "&&"))  # no mnemonic
            self.ffmpeg_group.setTitle(tr("ffmpeg_section"))
            self.microphone_group.setTitle(tr("microphone_group_title"))
            self._retranslate_modern_prompts_page()
            for row in self._rows:
                row.sync_texts()

    # --- Pages of the modern design ----------------------------------------------------
    #: (page, icon, title key, subtitle key) in sidebar order; empty in the classic design.
    _nav_pages: List[Tuple[QWidget, str, str, str]] = []

    def _build_modern_pages(self) -> None:
        """Split the long API providers and General tabs, so every page fits without scrolling."""
        rephrasing = self._scroll_page([self.shared_api_group])
        recording = self._scroll_page([self.hotkeys_group, self.recording_group])
        advanced = self._scroll_page([self.text_insertion_group, self.logging_group, self.ffmpeg_group])
        # The smaller pages have room to show their sections open.
        for section in (self.transcription_prompt_section, self.second_pass_section):
            section.set_expanded(True)
        self._nav_pages = [
            (self.transcription_tab, "mic", "page_transcription_title", "page_transcription_subtitle"),
            (rephrasing, "pen", "page_rephrasing_title", "page_rephrasing_subtitle"),
            (self._api_keys_tab, "key", "tab_api_keys", "tooltip_tab_api_keys"),
            (recording, "keyboard", "page_recording_title", "page_recording_subtitle"),
            (self.post_rephrasing_tab, "sparkles", "tab_transformations", "tooltip_tab_transformations"),
            (self._replacements_tab, "replace", "tab_replacements", "tooltip_tab_replacements"),
            (self.general_tab, "sliders", "tab_general", "page_general_subtitle"),
            (advanced, "code", "page_advanced_title", "page_advanced_subtitle"),
        ]
        # The existing tabs are already in this relative order; the new pages slot in between.
        for index, (page, _icon, _title, _subtitle) in enumerate(self._nav_pages):
            if self.tabs.indexOf(page) < 0:
                self.tabs.insertTab(index, page, "")
            self._nav.addItem(QListWidgetItem())
        self._nav.setCurrentRow(self.tabs.currentIndex())
        self._apply_design_icons(True, self.theme_palette())

    def _scroll_page(self, widgets: List[QWidget]) -> QWidget:
        """A page that scrolls ``widgets`` with the margins and spacing of the pages from the .ui file."""
        page = QWidget(self)
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        area = QScrollArea(page)
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget(area)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(self.transcription_layout.contentsMargins())
        layout.setSpacing(self.transcription_layout.spacing())
        for widget in widgets:
            layout.addWidget(widget)  # reparents it out of its old page
        layout.addStretch(1)
        area.setWidget(content)
        outer.addWidget(area)
        return page

    def fit_height_to_pages(self) -> None:
        """Grow the window so the tallest page shows without scrolling (up to the screen's height)."""
        if not self._modern:
            return
        self.ensurePolished()
        content = max((self._unscrolled_height(self.tabs.widget(index)) for index in range(self.tabs.count())),
                      default=0)
        margins = self._page_area.contentsMargins()
        chrome = (self._page_header.sizeHint().height() + self.footer.sizeHint().height()
                  + margins.top() + margins.bottom() + 2 * self._page_area.spacing())
        needed = content + chrome
        screen = self.screen()
        if screen is not None:
            needed = min(needed, screen.availableGeometry().height() - _SCREEN_HEADROOM)
        if self.height() < needed:
            self.resize(self.width(), needed)

    @staticmethod
    def _unscrolled_height(page: Optional[QWidget]) -> int:
        """The height ``page`` needs to show the whole content of its scroll areas."""
        if page is None:
            return 0
        height = page.sizeHint().height()
        for area in page.findChildren(QScrollArea):
            content = area.widget()
            if content is not None:
                height += max(0, content.sizeHint().height() - area.sizeHint().height())
        return height

    # --- Settings as rows ---------------------------------------------------------------
    _rows: List[SettingRow] = []
    _build_modern_prompts_page: Callable[[], None]
    _retranslate_modern_prompts_page: Callable[[], None]

    def _row(self, title: Optional[QLabel], *controls: object, **options: object) -> SettingRow:
        """A row of the modern card layout (see ``SettingRow``)."""
        row = SettingRow(title, controls, **options)  # type: ignore[arg-type]
        self._rows.append(row)
        return row

    def _switch(self, checkbox: QCheckBox) -> SettingRow:
        """A row whose title is the checkbox caption and whose control is its switch."""
        return self._row(None, checkbox, checkbox=checkbox)

    def _build_modern_rows(self) -> None:
        """Rebuild the cards of the API providers and General pages as rows (modern design, at startup)."""
        self._rows = []
        row, switch = self._row, self._switch
        for group, prefix_widgets, sections in (
            (self.transcription_api_group,
             (self.transcription_provider_label, self.transcription_provider_selector, self.model_label,
              self.model_dropdown, self.transcription_temp_label_title, self.transcription_temp_slider,
              self.transcription_temp_label, self.api_endpoint_label, self.api_endpoint_input, self.api_key_label,
              self.transcription_key_profile_selector, self.transcription_key_status_label,
              self.transcription_key_choose_button, self.transcription_key_add_button,
              self.test_transcription_api_button),
             (self.transcription_prompt_section, self.second_pass_section)),
            (self.shared_api_group,
             (self.rephrasing_provider_label, self.rephrasing_provider_selector, self.rephrasing_model_label,
              self.rephrasing_model_input, self.rephrasing_temp_label_title, self.rephrasing_temp_slider,
              self.rephrasing_temp_label, self.rephrasing_api_url_label, self.rephrasing_api_url_input,
              self.rephrasing_api_key_label, self.rephrasing_key_profile_selector, self.rephrasing_key_status_label,
              self.rephrasing_key_choose_button, self.rephrasing_key_add_button, self.test_rephrasing_api_button),
             ()),
        ):
            (provider_label, provider, model_label, model, temp_label, temp_slider, temp_value, url_label, url,
             key_label, key, status, choose, add, test) = prefix_widgets
            temp_value.setFixedWidth(34)
            temp_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._rebuild(group, [
                row(provider_label, provider), row(model_label, model),
                row(temp_label, temp_slider, temp_value, follow=temp_slider),
                row(url_label, url), row(key_label, key),
                row(status, choose, add, test, follow=status, plain_title=True),
                *sections,
            ])
        self._rebuild(self.second_pass_section.content, [
            row(self.second_pass_mode_label, self.second_pass_mode_selector),
            row(self.second_pass_logprob_label, self.second_pass_logprob_input),
            row(self.second_pass_prompt_label, self.second_pass_prompt_input),
        ], margins=(18, 0, 0, 4))
        # FFmpeg becomes a card like the others; its collapsible section stays behind, empty and hidden.
        self.ffmpeg_group = QGroupBox(self)
        QVBoxLayout(self.ffmpeg_group)
        self._rebuild(self.ffmpeg_group, [
            row(self.ffmpeg_status_label, plain_title=True, follow=self.ffmpeg_status_label),
            row(self.ffmpeg_label, self.ffmpeg_path_input, self.ffmpeg_browse_button),
        ])
        self.ffmpeg_path_input.setFixedWidth(220)
        self.ffmpeg_section.hide()

        for display in (self.hotkey_display, self.pr_hotkey_display):
            display.setMaximumWidth(180)
        self._rebuild(self.hotkeys_group, [
            row(self.hotkey_label, self.hotkey_display, self.set_hotkey_button),
            switch(self.push_to_talk_checkbox),
            row(self.pr_hotkey_label, self.pr_hotkey_display, self.set_pr_hotkey_button),
        ])
        self.gain_input.setMaximumWidth(110)
        self.recording_format_selector.setMaximumWidth(130)
        self.recording_bitrate_selector.setMaximumWidth(110)
        self._rebuild(self.recording_group, [
            row(self.input_language_label, self.language_input),
            row(self.gain_label, self.gain_input),
            switch(self.gain_normalize_checkbox),
            switch(self.windows_keep_mic_hot_checkbox),
            row(self.windows_keep_mic_hot_idle_label, self.windows_keep_mic_hot_idle_input),
            row(self.min_recording_label, self.min_recording_input),
            row(self.recording_format_label, self.recording_format_selector, self.recording_bitrate_label,
                self.recording_bitrate_selector, follow=self.recording_format_selector),
            switch(self.transcription_hedging_checkbox),
            row(self.transcription_hedging_wait_label, self.transcription_hedging_wait_input),
        ])

        # General page: the selectors above the cards get a card of their own.
        self.basics_group = QGroupBox(self.general_scroll_widget)
        QVBoxLayout(self.basics_group)
        clear_layout(self.general_selectors_layout)
        self.general_layout.removeItem(self.general_selectors_layout)
        self.general_layout.insertWidget(0, self.basics_group)
        self._rebuild(self.basics_group, [
            row(self.ui_language_label, self.ui_language_selector),
            row(self.color_theme_label, self.color_theme_selector),
            switch(self.modern_design_checkbox),
        ])
        # The microphone matters most where transcription is set up: a card at the top of that page.
        self.microphone_group = QGroupBox(self.transcription_api_group.parentWidget())
        QVBoxLayout(self.microphone_group)
        self.transcription_layout.insertWidget(0, self.microphone_group)
        self._rebuild(self.microphone_group, [row(self.input_device_label, self.input_device_selector)])
        self.proxy_url_input.setMaximumWidth(240)
        self._rebuild(self.misc_group, [
            row(self.proxy_url_label, self.proxy_url_input, self.test_internet_button),
            switch(self.use_px_proxy_checkbox),
            row(self.network_ip_family_label, self.network_ip_family_selector),
            switch(self.restore_clipboard_checkbox),
            switch(self.systray_double_click_copy_checkbox),
            switch(self.post_rephrase_auto_select_all_checkbox),
            switch(self.quit_without_confirmation_checkbox),
        ])
        self._rebuild(self.text_insertion_group, [
            switch(self.windows_sendinput_text_checkbox), switch(self.windows_sendinput_fallback_checkbox),
            switch(self.fast_paste_checkbox), switch(self.alt_clipboard_lib_checkbox),
        ])
        self._rebuild(self.logging_group, [
            row(self.log_retention_label, self.log_retention_input),
            switch(self.debug_logging_checkbox), switch(self.file_logging_checkbox),
            switch(self.redact_log_checkbox), switch(self.transcription_diagnostics_checkbox),
        ])
        self._build_modern_prompts_page()
        self._build_modern_pages()

    def _rebuild(self, owner: QWidget, rows: List[QWidget], **options: object) -> CardBody:
        """Put ``rows`` into ``owner``; a widget left behind would float at the card's corner, so hide it."""
        body = rebuild_card(owner, rows, **options)  # type: ignore[arg-type]
        for child in owner.findChildren(QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly):
            if child is not body and not child.isHidden():
                logging.warning("modern_rows: %s left outside the rows of %s", child.objectName() or child,
                                owner.objectName())
                child.hide()
        return body

    # --- Switching designs --------------------------------------------------------------
    def _apply_design(self, modern: bool, colors: Dict[str, str]) -> None:
        """Arrange the chrome for the design once; later calls (theme changes) only recolour the icons."""
        if not self._design_arranged:
            self._design_arranged = True
            self._modern = modern
            self._arrange_design(modern)
        self._apply_design_icons(modern, colors)

    def _arrange_design(self, modern: bool) -> None:
        """Show the modern chrome or the classic menu bar, branding header and tab bar."""
        self.setMinimumWidth(WINDOW_MIN_WIDTH + self.design_extra_width())
        self._sidebar.setVisible(modern)
        self._page_header.setVisible(modern)
        self.menu_bar.setVisible(not modern)
        self._brand_header.setVisible(not modern)
        tab_bar = self.tabs.tabBar()
        if tab_bar is not None:
            tab_bar.setVisible(not modern)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        if not modern:
            # The classic window kept its content 5 px above the bottom edge.
            self._column_layout.setContentsMargins(0, 0, 0, 5)
            self._page_area.setContentsMargins(0, 0, 0, 0)
            return
        self._column_layout.setContentsMargins(0, 0, 0, 0)
        self._column_layout.setSpacing(0)
        self._page_area.setContentsMargins(*_PAGE_MARGINS)
        self._page_area.setSpacing(6)
        self.footer_row.setContentsMargins(*_FOOTER_MARGINS)
        # The hotkey badge and the play button move from the hidden branding header into the sidebar.
        self._sidebar_badge.insertWidget(0, self._brand_hotkey_badge)
        self._sidebar_tools.addWidget(self.brand_play_button)
        self.brand_play_button.setObjectName("sidebarButton")

    def is_modern(self) -> bool:
        """Whether the window shows the modern design."""
        return self._modern

    def design_extra_width(self) -> int:
        """How much wider than its pages the window is in the current design (the stored width excludes it)."""
        return _EXTRA_WIDTH if self._modern else 0

    def _apply_design_icons(self, modern: bool, colors: Dict[str, str]) -> None:
        """Line icons in the theme's colours for the sidebar and its menu; none in the classic menu bar."""
        muted, selected = colors["muted"], colors.get("nav_selected", colors["accent"])
        for page, name, _title, _subtitle in self._nav_pages:
            item = self._nav.item(self.tabs.indexOf(page))
            if item is not None:
                item.setIcon(line_icon(name, muted, selected))
        menu_icon = colors.get("menu_icon", muted)
        for action, name in ((self.open_config_action, "file"), (self.open_log_file_action, "log"),
                             (self.about_action, "info"), (self.github_action, "code"),
                             (self.exit_action, "quit")):
            action.setIcon(line_icon(name, menu_icon) if modern else QIcon())
        self._more_button.setIcon(line_icon("more", menu_icon))

    def _playback_icon(self, active: bool) -> Optional[QIcon]:
        """Modern design: the play/stop line icon of the sidebar button (None in the classic design)."""
        if not self._modern:
            return None
        colors = self.theme_palette()
        return line_icon("stop" if active else "play", colors.get("menu_icon", colors["muted"]))
