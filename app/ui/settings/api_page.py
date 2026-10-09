"""API settings shared by the transcription and rephrasing pages: provider, key profile, model."""
from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Optional

from PyQt6.QtCore import QSignalBlocker
from PyQt6.QtWidgets import QComboBox, QLineEdit, QMessageBox

from app.core.api_keys import (
    PROVIDER_NAMES,
    TASK_KEY_FIELDS,
    key_format_warning,
    masked_api_key,
    provider_endpoint,
    provider_for_url,
    rephrasing_configured,
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
from app.ui.theme import set_style_state

API_TASKS = ("transcription", "rephrasing")


class _TaskWidgets(NamedTuple):
    """Settings controls that exist once per API task."""

    endpoint: QLineEdit
    provider: QComboBox
    key_profile: QComboBox
    model: QComboBox
    model_catalogs: Dict[str, List[str]]
    model_field: str


class ApiPage(SettingsWindowBase):
    """Endpoint/provider/key/model selection for both API tasks and their completeness state."""

    def _init_api_settings(self) -> None:
        """Fill provider/model/key selectors and keep them consistent while the form changes."""
        for task in API_TASKS:
            widgets = self._task_widgets(task)
            for provider in PROVIDER_NAMES:
                widgets.provider.addItem(self._provider_label(provider), provider)
            widgets.provider.setCurrentIndex(widgets.provider.findData(provider_for_url(widgets.endpoint.text())))
            widgets.provider.currentIndexChanged.connect(lambda _index, task=task: self._set_provider_endpoint(task))
            widgets.endpoint.textChanged.connect(lambda _url, task=task: self._refresh_model_selectors(only_task=task))
            widgets.endpoint.textChanged.connect(self._refresh_key_profile_selectors)
            widgets.key_profile.currentIndexChanged.connect(self._refresh_api_state)
        self._refresh_model_selectors(preserve_saved=True)
        self._refresh_key_profile_selectors()
        self._api_keys_tab.profiles_changed.connect(self._refresh_key_profile_selectors)
        self.model_dropdown.currentTextChanged.connect(lambda _text: self._update_prompt_token_counter())
        self.rephrasing_model_input.currentTextChanged.connect(self._refresh_api_state)
        for checkbox in (self.liveprompt_enabled_checkbox, self.generic_rephrase_enabled_checkbox):
            checkbox.stateChanged.connect(self._refresh_api_state)

    def _task_widgets(self, task: str) -> _TaskWidgets:
        """The endpoint/provider/key/model controls of one API task (transcription or rephrasing)."""
        if task == "transcription":
            return _TaskWidgets(self.api_endpoint_input, self.transcription_provider_selector,
                                self.transcription_key_profile_selector, self.model_dropdown,
                                TRANSCRIPTION_MODEL_OPTIONS, "model")
        return _TaskWidgets(self.rephrasing_api_url_input, self.rephrasing_provider_selector,
                            self.rephrasing_key_profile_selector, self.rephrasing_model_input,
                            REPHRASING_MODEL_OPTIONS, "rephrasing_model")

    def _provider_label(self, provider: str) -> str:
        """Display name of a provider; only the generic 'Custom' entry is translated."""
        return self.translator.tr("api_key_custom_provider") if provider == "custom" else PROVIDER_NAMES[provider]

    def _retranslate_api_settings(self) -> None:
        """Re-label the translated provider entry and the key selectors."""
        for task in API_TASKS:
            selector = self._task_widgets(task).provider
            selector.setItemText(selector.findData("custom"), self._provider_label("custom"))
        self._refresh_key_profile_selectors()

    def _set_provider_endpoint(self, task: str) -> None:
        """Apply a selected official endpoint, or clear the field for a custom URL."""
        widgets = self._task_widgets(task)
        widgets.endpoint.setText(provider_endpoint(widgets.provider.currentData(), task))

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
        """Select a valid profile on provider changes; preserve choices during profile edits."""
        profiles = self._api_keys_tab.profiles()
        for task in API_TASKS:
            widgets = self._task_widgets(task)
            selector = widgets.key_profile
            saved_id = self.config[TASK_KEY_FIELDS[task][1]]
            selected_id = selector.currentData()
            if selected_id is None:
                selected_id = saved_id
            provider = provider_for_url(widgets.endpoint.text())
            with QSignalBlocker(widgets.provider):
                widgets.provider.setCurrentIndex(widgets.provider.findData(provider))
            previous_provider = selector.property("key_provider")
            selector.setProperty("key_provider", provider)
            if previous_provider is not None and previous_provider != provider:
                # The endpoint switched provider: keep the saved key if it fits, else the first usable one.
                valid_ids = usable_profile_ids(profiles, provider)
                selected_id = saved_id if saved_id in valid_ids else next(iter(valid_ids), "")
            with QSignalBlocker(selector):
                selector.clear()
                selector.addItem(self.translator.tr("api_key_none"), "")
                for profile in profiles:
                    if profile["provider"] == provider:
                        label = f"{profile['name']} ({self._provider_label(provider)}) — {masked_api_key(profile['key'])}"
                        selector.addItem(label, profile["id"])
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
        """Mark incomplete API sections and enable temperature only where the model accepts it."""
        temperature_supported = rephrasing_supports_temperature(self.rephrasing_model_input.currentText())
        self.rephrasing_temp_slider.setEnabled(temperature_supported)
        self.rephrasing_temp_label.setEnabled(temperature_supported)
        form = self._form_api_config()
        set_style_state(self.transcription_api_group, "incomplete", not transcription_configured(form))
        rephrasing_incomplete = not rephrasing_configured(form)
        set_style_state(self.shared_api_group, "incomplete", rephrasing_incomplete)
        # Transformations use the same fields, so their unavailable state shows on that tab live.
        self.transformations_unavailable_label.setVisible(rephrasing_incomplete)

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
        """Store profiles, selections and models; False (nothing stored) if a profile lacks a name."""
        profiles = self._api_keys_tab.profiles()
        if any(not profile["name"] for profile in profiles):
            QMessageBox.warning(self, self.translator.tr("tab_api_keys"), self.translator.tr("api_key_name_required"))
            self.tabs.setCurrentWidget(self._api_keys_tab)
            return False
        self.config["api_key_profiles"] = profiles
        self.config["groq_key_rotation"] = self._api_keys_tab.groq_rotation.isChecked()
        for task in API_TASKS:
            widgets = self._task_widgets(task)
            self.config[TASK_KEY_FIELDS[task][1]] = widgets.key_profile.currentData() or ""
            self.config[widgets.model_field] = widgets.model.currentText().strip()
        return True
