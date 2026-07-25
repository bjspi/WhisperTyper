"""Reusable provider-account settings tab."""
from __future__ import annotations

from typing import Any, Dict, Mapping

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class ProviderAccountsWidget(QWidget):
    """Central key storage plus advanced feature-specific custom endpoints."""

    refresh_requested = pyqtSignal(str)
    model_catalog_ready = pyqtSignal(str, object)
    model_catalog_failed = pyqtSignal(str, str)

    def __init__(self, config: Mapping[str, Any], parent: QWidget | None = None) -> None:
        """Build the provider accounts form from persisted config."""
        super().__init__(parent)
        outer = QVBoxLayout(self)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget(scroll)
        self.content_layout = QVBoxLayout(content)

        self.intro_label = QLabel(content)
        self.intro_label.setWordWrap(True)
        self.content_layout.addWidget(self.intro_label)

        self.accounts_group = QGroupBox(content)
        accounts_layout = QFormLayout(self.accounts_group)
        self.key_inputs: Dict[str, QLineEdit] = {}
        self.refresh_buttons: Dict[str, QPushButton] = {}
        self.status_labels: Dict[str, QLabel] = {}
        keys = config.get("provider_api_keys", {})
        if not isinstance(keys, Mapping):
            keys = {}
        for provider, label in (("groq", "Groq"), ("openai", "OpenAI")):
            row = QWidget(self.accounts_group)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            key_input = QLineEdit(row)
            key_input.setEchoMode(QLineEdit.EchoMode.Password)
            key_input.setText(str(keys.get(provider, "") or ""))
            refresh_button = QPushButton(row)
            refresh_button.clicked.connect(
                lambda _checked=False, selected=provider: self.refresh_requested.emit(selected)
            )
            status = QLabel(row)
            status.setWordWrap(True)
            row_layout.addWidget(key_input, 1)
            row_layout.addWidget(refresh_button)
            row_layout.addWidget(status, 1)
            accounts_layout.addRow(f"{label}:", row)
            self.key_inputs[provider] = key_input
            self.refresh_buttons[provider] = refresh_button
            self.status_labels[provider] = status
        self.content_layout.addWidget(self.accounts_group)

        self.custom_group = QGroupBox(content)
        self.custom_group.setCheckable(True)
        self.custom_group.setChecked(False)
        custom_layout = QFormLayout(self.custom_group)
        self.custom_endpoint_inputs: Dict[str, QLineEdit] = {}
        self.custom_key_inputs: Dict[str, QLineEdit] = {}
        custom = config.get("custom_provider_settings", {})
        if not isinstance(custom, Mapping):
            custom = {}
        for feature in ("transcription", "rephrasing"):
            feature_config = custom.get(feature, {})
            if not isinstance(feature_config, Mapping):
                feature_config = {}
            endpoint = QLineEdit(self.custom_group)
            endpoint.setText(str(feature_config.get("endpoint", "") or ""))
            key_input = QLineEdit(self.custom_group)
            key_input.setEchoMode(QLineEdit.EchoMode.Password)
            key_input.setText(str(feature_config.get("api_key", "") or ""))
            custom_layout.addRow(QLabel(feature.title(), self.custom_group))
            custom_layout.addRow("Endpoint:", endpoint)
            custom_layout.addRow("API key:", key_input)
            self.custom_endpoint_inputs[feature] = endpoint
            self.custom_key_inputs[feature] = key_input
        self.content_layout.addWidget(self.custom_group)
        self.content_layout.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll)

    def provider_keys(self) -> Dict[str, str]:
        """Return current unsaved central keys."""
        return {provider: field.text().strip() for provider, field in self.key_inputs.items()}

    def custom_settings(self) -> Dict[str, Dict[str, str]]:
        """Return current unsaved feature-specific custom settings."""
        return {
            feature: {
                "endpoint": self.custom_endpoint_inputs[feature].text().strip(),
                "api_key": self.custom_key_inputs[feature].text().strip(),
            }
            for feature in self.custom_endpoint_inputs
        }

    def set_refreshing(self, provider: str) -> None:
        """Show a non-blocking catalog refresh state."""
        self.refresh_buttons[provider].setEnabled(False)
        self.status_labels[provider].setText("…")

    def finish_refresh(self, provider: str, message: str, success: bool) -> None:
        """Restore the refresh button and show the translated result."""
        self.refresh_buttons[provider].setEnabled(True)
        self.status_labels[provider].setText(("✅ " if success else "⚠️ ") + message)

    def retranslate(self, tr: Any) -> None:
        """Update all texts using the application's translation function."""
        self.intro_label.setText(tr("provider_accounts_intro"))
        self.accounts_group.setTitle(tr("provider_accounts_group"))
        self.custom_group.setTitle(tr("provider_custom_group"))
        self.custom_group.setToolTip(tr("provider_custom_tooltip"))
        for provider, button in self.refresh_buttons.items():
            button.setText(tr("provider_refresh_models"))
            button.setToolTip(tr("provider_refresh_models_tooltip", provider=provider.title()))
