"""API settings shared by the transcription and rephrasing pages: provider, model and key.

Both groups have the same controls. OpenAI and Groq use their official URL and, unless the
user picks another one, the provider's first usable key from the API Keys tab; only a custom
provider shows the URL field and the key selector.
"""
from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Optional

from PyQt6.QtCore import QSignalBlocker, Qt
from PyQt6.QtWidgets import QComboBox, QLabel, QLineEdit, QMessageBox, QPushButton

from app.core.api_keys import (
    PROVIDER_NAMES,
    TASK_CUSTOM_URL_FIELDS,
    TASK_KEY_FIELDS,
    key_format_warning,
    masked_api_key,
    provider_endpoint,
    provider_for_url,
    rephrasing_configured,
    resolve_key_profile,
    selected_api_key,
    transcription_configured,
    usable_profile_ids,
)
from app.core.models import (
    REPHRASING_MODEL_OPTIONS,
    TRANSCRIPTION_MODEL_OPTIONS,
    prompt_token_limit,
    rephrasing_supports_temperature,
)
from app.core.textutil import clean_model_name, estimate_tokens
from app.ui.settings.base import SettingsWindowBase
from app.ui.settings.widgets import show_temperature_support
from app.ui.theme import set_style_state

API_TASKS = ("transcription", "rephrasing")


class _TaskWidgets(NamedTuple):
    """Settings controls that exist once per API task."""

    endpoint_label: QLabel
    endpoint: QLineEdit
    provider: QComboBox
    key_label: QLabel
    key_profile: QComboBox
    key_status: QLabel
    key_choose: QPushButton
    key_add: QPushButton
    model: QComboBox
    model_catalogs: Dict[str, List[str]]
    model_field: str


class ApiPage(SettingsWindowBase):
    """Provider/model/key selection for both API tasks and their completeness state."""

    def _init_api_settings(self) -> None:
        """Fill provider/model/key selectors and keep them consistent while the form changes."""
        # The user's own URL per task, restored when switching back to the custom provider.
        self._custom_urls: Dict[str, str] = {}
        # Whether the key selector is shown for an official provider ("choose another key").
        self._key_choice_open: Dict[str, bool] = {}
        for task in API_TASKS:
            widgets = self._task_widgets(task)
            provider = provider_for_url(widgets.endpoint.text())
            for provider_id in PROVIDER_NAMES:
                widgets.provider.addItem(self._provider_label(provider_id), provider_id)
            widgets.provider.setCurrentIndex(widgets.provider.findData(provider))
            self._custom_urls[task] = (self.config.get(TASK_CUSTOM_URL_FIELDS[task], "")
                                       or (widgets.endpoint.text() if provider == "custom" else ""))
            # The selector opens on demand; the status line names the key in use either way.
            self._key_choice_open[task] = False
            for button in (widgets.key_choose, widgets.key_add):
                set_style_state(button, "link", True)
                button.setCursor(Qt.CursorShape.PointingHandCursor)
            widgets.provider.currentIndexChanged.connect(lambda _index, task=task: self._on_provider_changed(task))
            widgets.endpoint.textChanged.connect(lambda _url, task=task: self._on_endpoint_edited(task))
            widgets.key_profile.currentIndexChanged.connect(self._refresh_api_state)
            widgets.key_choose.clicked.connect(lambda _checked=False, task=task: self._open_key_choice(task))
            widgets.key_add.clicked.connect(lambda _checked=False: self.tabs.setCurrentWidget(self._api_keys_tab))
        self._refresh_model_selectors(preserve_saved=True)
        self._refresh_key_profile_selectors()
        self._api_keys_tab.profiles_changed.connect(self._refresh_key_profile_selectors)
        self.model_dropdown.currentTextChanged.connect(lambda _text: self._update_prompt_token_counter())
        self.rephrasing_model_input.currentTextChanged.connect(self._refresh_api_state)

    def _task_widgets(self, task: str) -> _TaskWidgets:
        """The controls of one API task (transcription or rephrasing)."""
        if task == "transcription":
            return _TaskWidgets(self.api_endpoint_label, self.api_endpoint_input, self.transcription_provider_selector,
                                self.api_key_label, self.transcription_key_profile_selector,
                                self.transcription_key_status_label, self.transcription_key_choose_button,
                                self.transcription_key_add_button, self.model_dropdown,
                                TRANSCRIPTION_MODEL_OPTIONS, "model")
        return _TaskWidgets(self.rephrasing_api_url_label, self.rephrasing_api_url_input, self.rephrasing_provider_selector,
                            self.rephrasing_api_key_label, self.rephrasing_key_profile_selector,
                            self.rephrasing_key_status_label, self.rephrasing_key_choose_button,
                            self.rephrasing_key_add_button, self.rephrasing_model_input,
                            REPHRASING_MODEL_OPTIONS, "rephrasing_model")

    def _provider_label(self, provider: str) -> str:
        """Display name of a provider; only the generic 'Custom' entry is translated."""
        return self.translator.tr("api_key_custom_provider") if provider == "custom" else PROVIDER_NAMES[provider]

    def _retranslate_api_settings(self) -> None:
        """Re-label the translated provider entry, the key selectors and the key status."""
        for task in API_TASKS:
            selector = self._task_widgets(task).provider
            selector.setItemText(selector.findData("custom"), self._provider_label("custom"))
        self._refresh_key_profile_selectors()

    def _on_provider_changed(self, task: str) -> None:
        """Apply the official URL or the user's own one; the key returns to the automatic choice."""
        widgets = self._task_widgets(task)
        self._key_choice_open[task] = False
        provider = widgets.provider.currentData()
        url = self._custom_urls[task] if provider == "custom" else provider_endpoint(provider, task)
        if widgets.endpoint.text() == url:
            self._on_endpoint_edited(task)  # same text emits no signal, but the controls must follow
        else:
            widgets.endpoint.setText(url)

    def _on_endpoint_edited(self, task: str) -> None:
        """Remember a custom URL and re-filter models and keys for the endpoint's provider."""
        url = self._task_widgets(task).endpoint.text()
        if provider_for_url(url) == "custom":
            self._custom_urls[task] = url
        self._refresh_model_selectors(only_task=task)
        self._refresh_key_profile_selectors()

    def _open_key_choice(self, task: str) -> None:
        """Show the key selector for an official provider, too."""
        self._key_choice_open[task] = True
        self._refresh_api_state()

    def _refresh_model_selectors(self, preserve_saved: bool = False, only_task: Optional[str] = None) -> None:
        """Filter models by endpoint; keep saved/custom names and replace incompatible built-ins on a provider change."""
        for task in API_TASKS:
            if only_task is not None and task != only_task:
                continue
            widgets = self._task_widgets(task)
            selector, catalogs = widgets.model, widgets.model_catalogs
            model = self.config[widgets.model_field] if preserve_saved else selector.currentText()
            options = catalogs.get(provider_for_url(widgets.endpoint.text()), [])
            model_id = clean_model_name(model)
            matching = next((option for option in options if clean_model_name(option) == model_id), None)
            known = any(clean_model_name(option) == model_id for models in catalogs.values() for option in models)
            if not preserve_saved and matching is None and known and options:
                model = options[0]
                matching = model
            with QSignalBlocker(selector):
                selector.clear()
                selector.addItems(options)
                selector.setCurrentText(matching or model)
        self._update_prompt_token_counter()
        self._refresh_api_state()

    def _refresh_key_profile_selectors(self) -> None:
        """List the provider's profiles behind "Automatic"; a provider change resets to automatic."""
        profiles = self._api_keys_tab.profiles()
        for task in API_TASKS:
            widgets = self._task_widgets(task)
            selector = widgets.key_profile
            selected_id = selector.currentData()
            if selected_id is None:
                selected_id = self.config[TASK_KEY_FIELDS[task][1]]
            provider = provider_for_url(widgets.endpoint.text())
            with QSignalBlocker(widgets.provider):
                widgets.provider.setCurrentIndex(widgets.provider.findData(provider))
            previous_provider = selector.property("key_provider")
            selector.setProperty("key_provider", provider)
            if previous_provider is not None and previous_provider != provider:
                selected_id = ""
            with QSignalBlocker(selector):
                selector.clear()
                selector.addItem(self.translator.tr("api_key_automatic"), "")
                for profile in profiles:
                    if profile["provider"] == provider:
                        selector.addItem(f"{profile['name']} — {masked_api_key(profile['key'])}", profile["id"])
                selector.setCurrentIndex(max(0, selector.findData(selected_id)))
        self._refresh_api_state()

    def _form_api_config(self) -> Dict[str, Any]:
        """Config snapshot of the unsaved API fields, so form checks reuse the runtime rules."""
        return {
            "api_key_profiles": self._api_keys_tab.profiles(),
            "api_endpoint": self.api_endpoint_input.text(),
            "transcription_key_profile_id": self.transcription_key_profile_selector.currentData(),
            "rephrasing_api_url": self.rephrasing_api_url_input.text(),
            "rephrasing_key_profile_id": self.rephrasing_key_profile_selector.currentData(),
            "rephrasing_model": self.rephrasing_model_input.currentText(),
        }

    def _ui_api_key(self, task: str) -> str:
        """Resolve credentials from unsaved form values for highlighting and connection tests."""
        return selected_api_key(self._form_api_config(), task)

    def _refresh_api_state(self, *_args: object) -> None:
        """Show the controls each provider needs, the key in use, and mark incomplete sections."""
        show_temperature_support(self.rephrasing_temp_slider, self.rephrasing_temp_label,
                                 rephrasing_supports_temperature(self.rephrasing_model_input.currentText()),
                                 self.translator)
        form = self._form_api_config()
        for task in API_TASKS:
            self._refresh_key_controls(task, form)
        set_style_state(self.transcription_api_group, "incomplete", not transcription_configured(form))
        rephrasing_incomplete = not rephrasing_configured(form)
        set_style_state(self.shared_api_group, "incomplete", rephrasing_incomplete)
        # Prompts use the same fields, so their unavailable state shows on that tab live.
        self.transformations_unavailable_label.setVisible(rephrasing_incomplete)

    def _refresh_key_controls(self, task: str, form: Dict[str, Any]) -> None:
        """URL/key visibility and the status line for one task."""
        widgets = self._task_widgets(task)
        provider = provider_for_url(widgets.endpoint.text())
        custom = provider == "custom"
        widgets.endpoint_label.setVisible(custom)
        widgets.endpoint.setVisible(custom)
        show_selector = custom or self._key_choice_open.get(task, False)
        widgets.key_label.setVisible(show_selector)
        widgets.key_profile.setVisible(show_selector)
        profile = resolve_key_profile(form, task)
        if profile is not None:
            widgets.key_status.setText(self.translator.tr(
                "api_key_status_ok", name=profile["name"], masked=masked_api_key(profile["key"])))
        else:
            widgets.key_status.setText(self.translator.tr(
                "api_key_status_missing", provider=self._provider_label(provider)))
        set_style_state(widgets.key_status, "key_status", "ok" if profile is not None else "missing")
        usable = usable_profile_ids(form["api_key_profiles"], provider)
        widgets.key_choose.setVisible(not show_selector and len(usable) > 1)
        widgets.key_add.setVisible(profile is None)

    def _collect_validation_warnings(self, model_raw: str) -> List[str]:
        """Translated hints about the current form (saving proceeds regardless)."""
        warnings: List[str] = []
        warning_key = key_format_warning(provider_for_url(self.api_endpoint_input.text()), self._ui_api_key("transcription"))
        if warning_key:
            warnings.append(self.translator.tr(warning_key))
        limit = prompt_token_limit(model_raw)
        if limit and estimate_tokens(self.prompt_input.toPlainText()) > limit:
            warnings.append(self.translator.tr("validation_prompt_too_long", limit=limit))
        return warnings

    def _save_api_settings(self) -> bool:
        """Store profiles, key choices, custom URLs and models; False (nothing stored) if a profile lacks a name."""
        profiles = self._api_keys_tab.profiles()
        if any(not profile["name"] for profile in profiles):
            QMessageBox.warning(self, self.translator.tr("tab_api_keys"), self.translator.tr("api_key_name_required"))
            self.tabs.setCurrentWidget(self._api_keys_tab)
            return False
        self.config["api_key_profiles"] = profiles
        self.config["groq_key_rotation"] = self._api_keys_tab.groq_rotation.isChecked()
        for task in API_TASKS:
            widgets = self._task_widgets(task)
            chosen = widgets.key_profile.currentData() or ""
            usable = usable_profile_ids(profiles, provider_for_url(widgets.endpoint.text()))
            if usable and chosen == usable[0]:
                # Same key the automatic choice picks: store "automatic", so it follows a later reordering.
                chosen = ""
                widgets.key_profile.setCurrentIndex(0)
            self.config[TASK_KEY_FIELDS[task][1]] = chosen
            self.config[TASK_CUSTOM_URL_FIELDS[task]] = self._custom_urls[task]
            self.config[widgets.model_field] = widgets.model.currentText().strip()
            self._key_choice_open[task] = False
        self._refresh_api_state()
        return True
