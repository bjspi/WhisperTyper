"""Central editor for named API credentials; secret values are always masked."""
from __future__ import annotations

import uuid
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core.api_keys import PROVIDER_NAMES
from app.core.i18n import TranslationManager


class ApiKeysTab(QWidget):
    """Keep unsaved profiles in the table and notify the task selectors of edits."""

    profiles_changed = pyqtSignal()

    def __init__(self, profiles: list[dict[str, str]], translator: TranslationManager, parent: Optional[QWidget] = None) -> None:
        """Build the editor without storing secrets in table labels or tooltips."""
        super().__init__(parent)
        self.translator = translator
        layout = QVBoxLayout(self)
        self.description = QLabel(self)
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        self.table = QTableWidget(0, 3, self)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        vertical_header, horizontal_header = self.table.verticalHeader(), self.table.horizontalHeader()
        assert vertical_header is not None and horizontal_header is not None
        vertical_header.hide()
        vertical_header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        horizontal_header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        buttons = QHBoxLayout()
        self.add_button = QPushButton(self)
        self.remove_button = QPushButton(self)
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.remove_button)
        buttons.addStretch()
        layout.addLayout(buttons)
        self.groq_rotation = QCheckBox(self)
        layout.addWidget(self.groq_rotation)
        for profile in profiles:
            self._append_profile(profile)
        self.table.itemChanged.connect(lambda _item: self.profiles_changed.emit())
        self.table.itemSelectionChanged.connect(lambda: self.remove_button.setEnabled(self.table.currentRow() >= 0))
        self.add_button.clicked.connect(self._add_profile)
        self.remove_button.clicked.connect(self._remove_profile)
        self.remove_button.setEnabled(False)
        self.retranslate_ui()

    def profiles(self) -> list[dict[str, str]]:
        """Return a fresh snapshot for saving or resolving the currently edited keys."""
        profiles = []
        for row in range(self.table.rowCount()):
            name = self.table.item(row, 0)
            provider = self.table.cellWidget(row, 1)
            secret = self.table.cellWidget(row, 2)
            assert name is not None and isinstance(provider, QComboBox) and isinstance(secret, QLineEdit)
            profiles.append({"id": name.data(Qt.ItemDataRole.UserRole), "name": name.text().strip(),
                             "provider": provider.currentData(), "key": secret.text().strip()})
        return profiles

    def _append_profile(self, profile: dict[str, str]) -> None:
        """Append a complete row before exposing its signals to the task selectors."""
        self.table.blockSignals(True)
        row = self.table.rowCount()
        self.table.insertRow(row)
        name = QTableWidgetItem(profile["name"])
        name.setData(Qt.ItemDataRole.UserRole, profile["id"])
        self.table.setItem(row, 0, name)
        provider = QComboBox(self.table)
        for provider_id, label in PROVIDER_NAMES.items():
            provider.addItem(label, provider_id)
        provider.setCurrentIndex(provider.findData(profile["provider"]))
        self.table.setCellWidget(row, 1, provider)
        secret = QLineEdit(profile["key"], self.table)
        secret.setEchoMode(QLineEdit.EchoMode.Password)
        self.table.setCellWidget(row, 2, secret)
        provider.currentIndexChanged.connect(lambda _index: self.profiles_changed.emit())
        secret.textChanged.connect(lambda _text: self.profiles_changed.emit())
        self.table.blockSignals(False)

    def _add_profile(self) -> None:
        """Create a named empty profile for the user to fill in."""
        self._append_profile({"id": uuid.uuid4().hex, "name": f"OpenAI {self.table.rowCount() + 1}", "provider": "openai", "key": ""})
        self.table.selectRow(self.table.rowCount() - 1)
        self.retranslate_ui()
        self.profiles_changed.emit()

    def _remove_profile(self) -> None:
        """Remove the selected row; existing task selections will become unselected."""
        if self.table.currentRow() >= 0:
            self.table.removeRow(self.table.currentRow())
            self.profiles_changed.emit()

    def retranslate_ui(self) -> None:
        """Translate labels without changing profile names or revealing credentials."""
        tr = self.translator.tr
        self.description.setText(tr("api_key_profiles_description"))
        self.table.setHorizontalHeaderLabels([tr("api_key_name"), tr("api_key_provider"), tr("api_key_label")])
        self.add_button.setText(tr("api_key_add"))
        self.remove_button.setText(tr("api_key_remove"))
        self.groq_rotation.setText(tr("groq_key_rotation"))
        self.groq_rotation.setToolTip(tr("groq_key_rotation_tooltip"))
        for row in range(self.table.rowCount()):
            provider = self.table.cellWidget(row, 1)
            secret = self.table.cellWidget(row, 2)
            assert isinstance(provider, QComboBox) and isinstance(secret, QLineEdit)
            provider.setItemText(provider.findData("custom"), tr("api_key_custom_provider"))
            secret.setAccessibleName(tr("api_key_label"))
