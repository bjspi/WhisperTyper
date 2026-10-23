"""Settings page of the local transcription engine: engine, server for other programs, test and benchmark.

The form values are saved with the settings window's Save (``load``/``save``); the engine
actions (load, unload, self-test, test, benchmark) act at once on the saved settings.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import QProcess, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.context import AppContext
from app.controllers.file_actions import ENGINE_LOG_PATH
from app.controllers.local_engine import UNAVAILABLE, LocalEngineController
from app.core.constants import APP_DATA_DIR
from app.core.i18n import TranslationManager
from app.local_engine import availability
from app.local_engine.catalog import (
    AUTO_DEVICE,
    CTRANSLATE2,
    DEFAULT_MODELS,
    DEFAULT_SERVER_PORT,
    DEVICES,
    LOAD_STRATEGIES,
    MODEL_CONFIG_KEYS,
    MODELS,
    backend_of,
    default_models_dir,
    model_dir_name,
    model_downloaded,
    model_path,
    resolve_device,
)
from app.local_engine.supervisor import CONFLICT, DOWN, LOADING, OFF, READY, STARTING, EngineStatus
from app.services.http_transport import request
from app.ui import message_dialog
from app.ui.local_engine_install import install
from app.ui.settings.modern_rows import SettingRow, rebuild_card
from app.ui.theme import set_style_state

#: Status colour per state (styled by the theme through ``level``).
_STATUS_LEVEL = {READY: "good", STARTING: "busy", LOADING: "busy", OFF: "off", DOWN: "bad", CONFLICT: "bad",
                 UNAVAILABLE: "bad"}


def _megabytes(size: int) -> str:
    return f"{size / (1024 * 1024):,.0f} MB" if size < 1024 ** 3 else f"{size / 1024 ** 3:.1f} GB"


class LocalEngineTab(QWidget):
    """Cards of the local engine page; edits wait for Save, actions run at once."""

    #: The "enabled" switch changed in the form (the transcription page shows whether "Local" can work).
    enabled_changed = pyqtSignal(bool)
    _test_done = pyqtSignal(object)

    def __init__(self, ctx: AppContext, engine: LocalEngineController, translator: TranslationManager,
                 parent: Optional[QWidget] = None) -> None:
        """Build the cards and load the saved settings into them."""
        super().__init__(parent)
        self._ctx = ctx
        self._engine = engine
        self.translator = translator
        self._rows: List[SettingRow] = []
        #: The chosen model per backend while the form is open.
        self._models: Dict[str, str] = {}
        self._bench: Optional[QProcess] = None
        self._bench_buffer = ""
        self._loading_form = False
        self._saved_device = AUTO_DEVICE

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        area = QScrollArea(self)
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget(area)
        self._layout = QVBoxLayout(content)
        area.setWidget(content)
        outer.addWidget(area)
        self._build_engine_card()
        self._build_server_card()
        self._build_test_card()
        self._layout.addStretch(1)

        engine.status_changed.connect(self._render_status)
        engine.devices_changed.connect(self._refresh_devices)
        self._test_done.connect(self._on_test_done)
        self.load(ctx.config)
        self.retranslate_ui()

    # --- Building --------------------------------------------------------------------------
    def _row(self, title: Optional[QLabel], *controls: Any, **options: Any) -> SettingRow:
        row = SettingRow(title, controls, **options)
        self._rows.append(row)
        return row

    def _card(self, rows: List[QWidget]) -> QGroupBox:
        group = QGroupBox(self)
        QVBoxLayout(group)
        rebuild_card(group, rows)
        self._layout.addWidget(group)
        return group

    def _build_engine_card(self) -> None:
        self.enabled_checkbox = QCheckBox(self)
        self.status_label = QLabel(self)
        self.status_label.setObjectName("local_engine_status")
        self.status_label.setWordWrap(True)
        self.load_button = QPushButton(self)
        self.restart_button = QPushButton(self)
        self.self_test_button = QPushButton(self)
        self.packages_label = QLabel(self)
        self.packages_label.setObjectName("local_engine_status")
        self.packages_label.setWordWrap(True)
        self.install_button = QPushButton(self)
        self.device_label = QLabel(self)
        self.device_selector = QComboBox(self)
        self.detect_button = QPushButton(self)
        self.model_label = QLabel(self)
        self.model_selector = QComboBox(self)
        self.delete_model_button = QPushButton(self)
        self.models_dir_label = QLabel(self)
        self.models_dir_input = QLineEdit(self)
        self.models_dir_button = QPushButton(self)
        self.strategy_label = QLabel(self)
        self.strategy_selector = QComboBox(self)
        for strategy in LOAD_STRATEGIES:
            self.strategy_selector.addItem("", strategy)
        self.idle_label = QLabel(self)
        self.idle_input = QSpinBox(self)
        self.idle_input.setRange(0, 1440)
        self.vad_checkbox = QCheckBox(self)
        self.models_dir_input.setMaximumWidth(220)
        self.engine_group = self._card([
            self._row(None, self.enabled_checkbox, checkbox=self.enabled_checkbox),
            self._row(self.status_label, self.load_button, self.restart_button, self.self_test_button,
                      follow=self.status_label, plain_title=True),
            self._row(self.packages_label, self.install_button, follow=self.packages_label, plain_title=True),
            self._row(self.device_label, self.device_selector, self.detect_button, follow=self.device_selector),
            self._row(self.model_label, self.model_selector, self.delete_model_button, follow=self.model_selector),
            self._row(self.models_dir_label, self.models_dir_input, self.models_dir_button,
                      follow=self.models_dir_input),
            self._row(self.strategy_label, self.strategy_selector),
            self._row(self.idle_label, self.idle_input),
            self._row(None, self.vad_checkbox, checkbox=self.vad_checkbox),
        ])
        self.enabled_checkbox.toggled.connect(self._on_enabled_toggled)
        self.load_button.clicked.connect(self._toggle_loaded)
        self.restart_button.clicked.connect(self._engine.restart)
        self.self_test_button.clicked.connect(self._engine.self_test)
        self.install_button.clicked.connect(self._install_packages)
        self.detect_button.clicked.connect(self._engine.detect_devices)
        self.device_selector.currentIndexChanged.connect(lambda _index: self._on_device_changed())
        self.model_selector.currentIndexChanged.connect(lambda _index: self._on_model_changed())
        self.delete_model_button.clicked.connect(self._delete_model)
        self.models_dir_button.clicked.connect(self._browse_models_dir)
        self.models_dir_input.editingFinished.connect(self._refresh_models)

    def _build_server_card(self) -> None:
        self.server_checkbox = QCheckBox(self)
        self.host_label = QLabel(self)
        self.host_input = QLineEdit(self)
        self.host_input.setMaximumWidth(180)
        self.port_label = QLabel(self)
        self.port_input = QSpinBox(self)
        self.port_input.setRange(1024, 65535)
        self.endpoint_label = QLabel(self)
        self.endpoint_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.endpoint_label.setWordWrap(True)
        self.copy_endpoint_button = QPushButton(self)
        self.health_label = QLabel(self)
        self.health_label.setWordWrap(True)
        self.server_group = self._card([
            self._row(None, self.server_checkbox, checkbox=self.server_checkbox),
            self._row(self.host_label, self.host_input),
            self._row(self.port_label, self.port_input),
            self._row(self.endpoint_label, self.copy_endpoint_button, follow=self.endpoint_label, plain_title=True),
            self._row(self.health_label, follow=self.health_label, plain_title=True),
        ])
        self.server_checkbox.toggled.connect(lambda _on: self._update_enabled_state())
        self.host_input.textChanged.connect(lambda _text: self._update_endpoint())
        self.port_input.valueChanged.connect(lambda _value: self._update_endpoint())
        self.copy_endpoint_button.clicked.connect(self._copy_endpoint)

    def _build_test_card(self) -> None:
        self.test_label = QLabel(self)
        self.test_button = QPushButton(self)
        self.test_result_label = QLabel(self)
        self.test_result_label.setWordWrap(True)
        self.test_result_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.bench_label = QLabel(self)
        self.bench_button = QPushButton(self)
        self.bench_file_button = QPushButton(self)
        self.bench_status_label = QLabel(self)
        self.bench_status_label.setWordWrap(True)
        self.bench_table = QTableWidget(0, 5, self)
        self.bench_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.bench_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.bench_table.setShowGrid(False)
        self.bench_table.setFixedHeight(150)
        vertical, horizontal = self.bench_table.verticalHeader(), self.bench_table.horizontalHeader()
        assert vertical is not None and horizontal is not None
        vertical.hide()
        horizontal.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        horizontal.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.bench_table.hide()
        self.log_label = QLabel(self)
        self.log_button = QPushButton(self)
        self.test_group = self._card([
            self._row(self.test_label, self.test_button),
            self._row(self.test_result_label, follow=self.test_result_label, plain_title=True),
            self._row(self.bench_label, self.bench_button, self.bench_file_button),
            self._row(self.bench_status_label, follow=self.bench_status_label, plain_title=True),
            self._row(None, self.bench_table, follow=self.bench_table, stacked=True),
            self._row(self.log_label, self.log_button),
        ])
        self.test_result_label.hide()
        self.bench_status_label.hide()
        self.test_button.clicked.connect(self._test_last_recording)
        self.bench_button.clicked.connect(lambda: self._start_benchmark(None))
        self.bench_file_button.clicked.connect(self._pick_benchmark_file)
        self.log_button.clicked.connect(self._open_log)

    # --- Form ------------------------------------------------------------------------------
    def load(self, config: Dict[str, Any]) -> None:
        """Fill the form from the saved settings."""
        self._loading_form = True
        try:
            self._saved_device = str(config.get("local_device") or AUTO_DEVICE)
            # Unsupported here (macOS, frozen build): the switch stays off and cannot be turned on.
            self.enabled_checkbox.setEnabled(self._engine.supported())
            self.enabled_checkbox.setChecked(self._engine.supported()
                                             and bool(config.get("local_engine_enabled", False)))
            self._models = {backend: str(config.get(key) or DEFAULT_MODELS[backend])
                            for backend, key in MODEL_CONFIG_KEYS.items()}
            self.models_dir_input.setText(str(config.get("local_models_dir", "")))
            self.models_dir_input.setPlaceholderText(str(default_models_dir(APP_DATA_DIR)))
            self.strategy_selector.setCurrentIndex(max(0, self.strategy_selector.findData(
                config.get("local_load_strategy", "preload"))))
            self.idle_input.setValue(int(config.get("local_idle_unload_minutes", 0) or 0))
            self.vad_checkbox.setChecked(bool(config.get("local_vad", True)))
            self.server_checkbox.setChecked(bool(config.get("local_server_enabled", False)))
            self.host_input.setText(str(config.get("local_server_host") or "127.0.0.1"))
            self.port_input.setValue(int(config.get("local_server_port") or DEFAULT_SERVER_PORT))
            self._refresh_devices()
        finally:
            self._loading_form = False
        self._update_enabled_state()
        self._render_status(self._engine.status)
        if self.enabled_checkbox.isChecked() and not self._engine.devices:
            self._engine.detect_devices()

    def save(self, config: Dict[str, Any]) -> None:
        """Write the form into ``config``."""
        config["local_engine_enabled"] = self.enabled_checkbox.isChecked()
        config["local_device"] = self.device_selector.currentData() or AUTO_DEVICE
        for backend, key in MODEL_CONFIG_KEYS.items():
            config[key] = self._models.get(backend) or DEFAULT_MODELS[backend]
        config["local_models_dir"] = self.models_dir_input.text().strip()
        config["local_load_strategy"] = self.strategy_selector.currentData()
        config["local_idle_unload_minutes"] = self.idle_input.value()
        config["local_vad"] = self.vad_checkbox.isChecked()
        config["local_server_enabled"] = self.server_checkbox.isChecked()
        config["local_server_host"] = self.host_input.text().strip() or "127.0.0.1"
        config["local_server_port"] = self.port_input.value()
        self._saved_device = config["local_device"]

    def is_enabled(self) -> bool:
        """The form's "enabled" switch (unsaved)."""
        return self.enabled_checkbox.isChecked()

    def request_enable(self) -> bool:
        """Turn the engine on (selecting the "Local" provider); False when the user declined the install."""
        if not self.enabled_checkbox.isChecked():
            self.enabled_checkbox.setChecked(True)
        return self.enabled_checkbox.isChecked()

    # --- Enabling and packages -------------------------------------------------------------
    def _on_enabled_toggled(self, on: bool) -> None:
        if self._loading_form:
            return  # ``load`` updates the page itself once all values are in
        if on and not self._ensure_packages():
            self.enabled_checkbox.setChecked(False)
            return
        if on and not self._engine.devices:
            self._engine.detect_devices()  # none yet, or an empty list from before the packages were installed
        self._update_enabled_state()
        self.enabled_changed.emit(on)

    def _missing_requirements(self) -> List[str]:
        return availability.install_packages(availability.display_adapters())

    def _ensure_packages(self) -> bool:
        """Offer to install what is missing; True when the engine can run afterwards."""
        if not self._engine.supported():
            return False
        requirements = self._missing_requirements()
        if not requirements:
            return True
        ok = install(self, self.translator, requirements)
        self._engine.detect_devices()
        self._update_packages_row()
        return ok and availability.engine_installed()

    def _install_packages(self) -> None:
        self._ensure_packages()

    def _update_packages_row(self) -> None:
        tr = self.translator.tr
        if not self._engine.supported():
            self.packages_label.setText(tr("local_engine_unsupported"))
            self.install_button.hide()
            self.packages_label.show()
            set_style_state(self.packages_label, "level", "bad")
            return
        requirements = self._missing_requirements()
        self.packages_label.setVisible(bool(requirements))
        self.install_button.setVisible(bool(requirements))
        if requirements:
            self.packages_label.setText(tr("local_engine_packages_hint", packages=", ".join(
                requirement.split(">")[0].split("<")[0] for requirement in requirements)))
            set_style_state(self.packages_label, "level", "busy")

    def _update_enabled_state(self) -> None:
        on = self.enabled_checkbox.isChecked()
        for widget in (self.load_button, self.restart_button, self.self_test_button, self.device_selector,
                       self.detect_button, self.model_selector, self.models_dir_input, self.models_dir_button,
                       self.strategy_selector, self.idle_input, self.vad_checkbox, self.server_checkbox,
                       self.test_button, self.bench_button, self.bench_file_button):
            widget.setEnabled(on)
        self.detect_button.setEnabled(on and not self._engine.detecting)
        server = on and self.server_checkbox.isChecked()
        # Address, port, endpoint and health check only matter (and show) while the server is on.
        server_rows: Tuple[QWidget, ...] = (self.host_input, self.port_input, self.endpoint_label, self.health_label)
        for row_control in server_rows:
            row_control.setVisible(self.server_checkbox.isChecked())
        for widget in (self.host_input, self.port_input, self.copy_endpoint_button):
            widget.setEnabled(server)
        # The server keeps the engine loaded: no idle unload then.
        self.idle_input.setEnabled(on and not server)
        self._update_delete_button()
        self._update_packages_row()
        self._update_endpoint()
        self._render_status(self._engine.status)

    # --- Devices and models ----------------------------------------------------------------
    def _device_text(self, device_id: str, name: str) -> str:
        return self.translator.tr(f"local_device_{device_id}", name=name)

    def _refresh_devices(self) -> None:
        """List the detected devices; a configured device that was not detected stays selectable."""
        tr = self.translator.tr
        current = self.device_selector.currentData() if self.device_selector.count() else self._saved_device
        detected = self._engine.devices or []
        auto = resolve_device(AUTO_DEVICE, [device["id"] for device in detected])
        auto_name = next((self._device_text(d["id"], d["name"]) for d in detected if d["id"] == auto), "")
        self.device_selector.blockSignals(True)
        self.device_selector.clear()
        detecting = self._engine.detecting
        self.detect_button.setText(tr("local_detect_running" if detecting else "local_detect_button"))
        self.detect_button.setEnabled(self.enabled_checkbox.isChecked() and not detecting)
        if self._engine.devices is None or detecting:
            self.device_selector.addItem(tr("local_device_auto_pending"), AUTO_DEVICE)
        else:
            self.device_selector.addItem(tr("local_device_auto", device=auto_name or tr("local_device_none")),
                                         AUTO_DEVICE)
        for device in detected:
            self.device_selector.addItem(self._device_text(device["id"], device["name"]), device["id"])
        if current and current != AUTO_DEVICE and self.device_selector.findData(current) < 0 and current in DEVICES:
            self.device_selector.addItem(tr("local_device_not_detected", device=self._device_text(current, "")),
                                         current)
        self.device_selector.setCurrentIndex(max(0, self.device_selector.findData(current)))
        self.device_selector.blockSignals(False)
        errors = "\n".join(f"{name}: {error}" for name, error in self._engine.device_errors.items())
        self.device_selector.setToolTip(errors)
        self._refresh_models()

    def _form_device(self) -> Optional[str]:
        choice = self.device_selector.currentData() or AUTO_DEVICE
        return resolve_device(choice, [device["id"] for device in self._engine.devices or []])

    def _form_backend(self) -> str:
        device = self._form_device()
        return backend_of(device) if device else CTRANSLATE2

    def _on_device_changed(self) -> None:
        self._refresh_models()

    def _models_dir(self) -> str:
        return os.path.expanduser(self.models_dir_input.text().strip() or str(default_models_dir(APP_DATA_DIR)))

    def _refresh_models(self) -> None:
        """The backend's models, marked when downloaded; a hand-entered model stays in the list."""
        tr = self.translator.tr
        backend = self._form_backend()
        selected = self._models.get(backend) or DEFAULT_MODELS[backend]
        models_dir = Path(self._models_dir())
        self.model_selector.blockSignals(True)
        self.model_selector.clear()
        repos = [model.repo for model in MODELS[backend]]
        for model in MODELS[backend]:
            parts = [f"{model.name} (~{_megabytes(model.size_mb * 1024 * 1024)})"]
            if model.recommended:
                parts.append(tr("local_model_recommended"))
            folder = model_path(models_dir, model.repo)
            if model_downloaded(backend, folder):
                parts.append(tr("local_model_downloaded"))
            elif folder.is_dir():
                parts.append(tr("local_model_partial"))
            self.model_selector.addItem(" · ".join(parts), model.repo)
        if selected not in repos:
            self.model_selector.addItem(selected, selected)
        self.model_selector.setCurrentIndex(max(0, self.model_selector.findData(selected)))
        self.model_selector.blockSignals(False)
        self._update_delete_button()

    def _on_model_changed(self) -> None:
        repo = self.model_selector.currentData()
        if repo:
            self._models[self._form_backend()] = repo
        self._update_delete_button()

    def _selected_model_folder(self) -> Optional[Path]:
        """Folder of the selected model when there is one (complete or a partial download)."""
        repo = self.model_selector.currentData()
        if not repo or Path(repo).is_absolute():
            return None  # a model folder of the user's own is never deleted from here
        folder = model_path(Path(self._models_dir()), repo)
        try:
            return folder if folder.is_dir() and any(folder.iterdir()) else None
        except OSError:
            return None

    def _update_delete_button(self) -> None:
        self.delete_model_button.setEnabled(self.enabled_checkbox.isChecked()
                                            and self._selected_model_folder() is not None)

    def _delete_model(self) -> None:
        """Delete the selected downloaded model (unloading the engine first when it uses it)."""
        tr = self.translator.tr
        folder = self._selected_model_folder()
        if folder is None:
            return
        size = sum(file.stat().st_size for file in folder.rglob("*") if file.is_file())
        confirmed = message_dialog.ask(self, tr("local_delete_model_title"),
                                       tr("local_delete_model_text", model=folder.name, size=_megabytes(size),
                                          path=str(folder)),
                                       tr("delete_button"), tr("dialog_cancel_button"), message_dialog.WARNING)
        if not confirmed:
            return
        status = self._engine.status
        in_use = status.state not in (OFF, UNAVAILABLE) and model_dir_name(status.model) == folder.name
        if in_use:
            # Windows keeps the files of a loaded model locked: unload, then delete once the process ended.
            logging.info("local_model_delete unloading the engine first: %s", folder)
            self._engine.stop()
            self.delete_model_button.setEnabled(False)
            QTimer.singleShot(4000, lambda: self._remove_model_folder(folder))
        else:
            self._remove_model_folder(folder)

    def _remove_model_folder(self, folder: Path) -> None:
        tr = self.translator.tr
        try:
            shutil.rmtree(folder)
            logging.info("local_model_deleted %s", folder)
            self._ctx.notifier.show(tr("local_delete_model_done", model=folder.name))
        except OSError as error:
            logging.error("local_model_delete failed %s: %s", folder, error)
            message_dialog.inform(self, tr("local_delete_model_title"), tr("local_delete_model_failed", error=error),
                                  tr("ok_button"), message_dialog.ERROR)
        self._refresh_models()

    def _browse_models_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, self.translator.tr("local_models_dir_dialog"),
                                                  self._models_dir())
        if folder:
            self.models_dir_input.setText(folder)
            self._refresh_models()

    # --- Server ----------------------------------------------------------------------------
    def _endpoint(self) -> str:
        host = self.host_input.text().strip() or "127.0.0.1"
        return f"http://{host}:{self.port_input.value()}/v1/audio/transcriptions"

    def _update_endpoint(self) -> None:
        tr = self.translator.tr
        self.endpoint_label.setText(tr("local_server_endpoint", endpoint=self._endpoint()))
        host = self.host_input.text().strip()
        exposed = host not in ("", "127.0.0.1", "localhost", "::1")
        self.host_input.setToolTip(tr("local_server_host_warning") if exposed else tr("local_server_host_tooltip"))
        set_style_state(self.host_input, "warning", exposed)

    def _copy_endpoint(self) -> None:
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self._endpoint())
            self._ctx.notifier.show(self.translator.tr("local_server_endpoint_copied"))

    # --- Status ----------------------------------------------------------------------------
    def status_text(self, status: EngineStatus) -> str:
        """One line for the status: state, device, model, progress or problem."""
        return status_text(self.translator, status, self._engine.devices or [])

    def _render_status(self, status: EngineStatus) -> None:
        tr = self.translator.tr
        self.status_label.setText(self.status_text(status))
        set_style_state(self.status_label, "level", _STATUS_LEVEL.get(status.state, "off"))
        running = status.state not in (OFF, UNAVAILABLE)
        self.load_button.setText(tr("local_unload_button" if running else "local_load_button"))
        self.self_test_button.setEnabled(self.enabled_checkbox.isChecked() and status.state == READY)
        if status.state == READY and status.checked_at:
            latency = tr("local_health_latency", seconds=f"{status.latency_s:.2f}") if status.latency_s else ""
            self.health_label.setText(tr("local_health_line", time=status.checked_at, latency=latency))
        else:
            self.health_label.setText(tr("local_health_none"))

    def _toggle_loaded(self) -> None:
        if self._engine.status.state not in (OFF, UNAVAILABLE):
            self._engine.stop()
        else:
            self._engine.restart()

    # --- Test ------------------------------------------------------------------------------
    def _test_last_recording(self) -> None:
        """Transcribe the newest recording with the running engine and show the text and the time."""
        tr = self.translator.tr
        latest = self._ctx.recordings.latest()
        self.test_result_label.show()
        if latest is None or not os.path.isfile(latest):
            self.test_result_label.setText(tr("no_recording_found_message"))
            return
        if self._engine.status.state != READY:
            self.test_result_label.setText(tr("local_test_not_ready"))
            return
        endpoint = self._engine.endpoint()
        language = str(self._ctx.config.get("input_language", ""))
        self.test_button.setEnabled(False)
        self.test_result_label.setText(tr("api_test_testing_button"))

        def run() -> None:
            t0 = time.perf_counter()
            try:
                with open(latest, "rb") as audio:
                    response = request("POST", endpoint, stage="local_engine_test", timeout=(30.0, 300.0),
                                       files={"file": (os.path.basename(latest), audio, "audio/wav")},
                                       data={"language": language})
                body = response.json() if response.status_code == 200 else {"error": response.text[:300]}
                result: Dict[str, Any] = {**body, "seconds": time.perf_counter() - t0}
            except Exception as error:  # noqa: BLE001 - shown on the page
                result = {"error": str(error)}
            try:
                self._test_done.emit(result)
            except RuntimeError:
                pass

        threading.Thread(target=run, name="LocalEngineTest", daemon=True).start()

    def _on_test_done(self, result: Dict[str, Any]) -> None:
        tr = self.translator.tr
        self.test_button.setEnabled(self.enabled_checkbox.isChecked())
        if "error" in result:
            logging.warning("local_engine_test failed: %s", result["error"])
            self.test_result_label.setText(tr("local_test_failed", error=result["error"]))
            return
        logging.info("local_engine_test ok seconds=%.2f chars=%d", result["seconds"], len(result.get("text", "")))
        self.test_result_label.setText(tr("local_test_result", seconds=f"{result['seconds']:.2f}",
                                          text=result.get("text", "") or tr("local_test_empty")))

    # --- Benchmark -------------------------------------------------------------------------
    def _pick_benchmark_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, self.translator.tr("local_bench_file_dialog"))
        if path:
            self._start_benchmark(path)

    def _start_benchmark(self, path: Optional[str]) -> None:
        """Time every downloaded model of the selected device on a recording (separate process)."""
        tr = self.translator.tr
        if self._bench is not None:
            self._bench.kill()
            return
        path = path or self._ctx.recordings.latest()
        device = self._form_device()
        self.bench_status_label.show()
        if not path or not os.path.isfile(path):
            self.bench_status_label.setText(tr("local_bench_no_audio"))
            return
        if device is None:
            self.bench_status_label.setText(tr("local_engine_no_device"))
            return
        self.bench_table.setRowCount(0)
        self.bench_table.show()
        process = QProcess(self)
        process.setProgram(availability.console_python())
        process.setArguments(["-m", "app.local_engine.bench", path, "--device", device, "--models-dir",
                              self._models_dir(), "--language", str(self._ctx.config.get("input_language", ""))])
        from app.local_engine.supervisor import PROJECT_ROOT

        process.setWorkingDirectory(PROJECT_ROOT)
        process.readyReadStandardOutput.connect(self._read_benchmark)
        process.finished.connect(lambda code, _status: self._on_benchmark_finished(code))
        self._bench, self._bench_buffer = process, ""
        self.bench_button.setText(tr("local_bench_stop"))
        self.bench_status_label.setText(tr("local_bench_running"))
        logging.info("local_engine_benchmark start device=%s file=%s", device, os.path.basename(path))
        process.start()

    def _read_benchmark(self) -> None:
        if self._bench is None:
            return
        self._bench_buffer += bytes(self._bench.readAllStandardOutput().data()).decode("utf-8", "replace")
        *lines, self._bench_buffer = self._bench_buffer.split("\n")
        for line in lines:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            self._on_benchmark_event(event)

    def _on_benchmark_event(self, event: Dict[str, Any]) -> None:
        tr = self.translator.tr
        kind = event.get("event")
        logging.info("local_engine_benchmark %s", {k: v for k, v in event.items() if k != "text"})
        if kind == "start":
            self.bench_status_label.setText(tr("local_bench_started", audio=event.get("audio_s"),
                                               models=event.get("models")))
        elif kind == "loading":
            self.bench_status_label.setText(tr("local_bench_loading", model=event.get("name")))
        elif kind in ("result", "error"):
            row = self.bench_table.rowCount()
            self.bench_table.insertRow(row)
            if kind == "result":
                cells = [event.get("name", ""), f"{event['load_s']:.2f} s", f"{event['first_s']:.2f} s",
                         f"{event['best_s']:.2f} s", f"{event['rtf']:.3f}" if event.get("rtf") is not None else "-"]
            else:
                cells = [event.get("name", ""), tr("local_bench_error"), "", "", ""]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setToolTip(event.get("text") or event.get("error") or "")
                self.bench_table.setItem(row, column, item)

    def _on_benchmark_finished(self, code: int) -> None:
        tr = self.translator.tr
        self._read_benchmark()
        if self._bench is not None:
            if code != 0:
                error = bytes(self._bench.readAllStandardError().data()).decode("utf-8", "replace").strip()
                logging.warning("local_engine_benchmark exit code %s: %s", code, error[-2000:])
                self.bench_status_label.setText(tr("local_bench_failed", error=error.splitlines()[-1] if error else code))
            else:
                self.bench_status_label.setText(tr("local_bench_done"))
            self._bench.deleteLater()
        self._bench = None
        self.bench_button.setText(tr("local_bench_button"))
        if self.bench_table.rowCount() == 0 and code == 0:
            self.bench_status_label.setText(tr("local_bench_no_models"))

    def _open_log(self) -> None:
        if os.path.isfile(ENGINE_LOG_PATH):
            QDesktopServices.openUrl(QUrl.fromLocalFile(ENGINE_LOG_PATH))
        else:
            self._ctx.notifier.show(self.translator.tr("log_file_not_exist_message"))

    def shutdown(self) -> None:
        """Stop a running benchmark (window closing)."""
        if self._bench is not None:
            self._bench.kill()

    # --- Translation -----------------------------------------------------------------------
    def retranslate_ui(self) -> None:
        """Apply the current UI language to every text of the page."""
        tr = self.translator.tr
        self.engine_group.setTitle(tr("local_engine_group"))
        self.server_group.setTitle(tr("local_server_group"))
        self.test_group.setTitle(tr("local_test_group"))
        self.enabled_checkbox.setText(tr("local_engine_enabled"))
        self.enabled_checkbox.setToolTip(tr("local_engine_enabled_tooltip"))
        self.restart_button.setText(tr("local_restart_button"))
        self.self_test_button.setText(tr("local_self_test_button"))
        self.self_test_button.setToolTip(tr("local_self_test_tooltip"))
        self.install_button.setText(tr("local_install_button"))
        self.device_label.setText(tr("local_device_label"))
        self.device_label.setToolTip(tr("local_device_tooltip"))
        self.detect_button.setText(tr("local_detect_button"))
        self.model_label.setText(tr("local_model_label"))
        self.model_label.setToolTip(tr("local_model_tooltip"))
        self.delete_model_button.setText(tr("local_delete_model_button"))
        self.delete_model_button.setToolTip(tr("local_delete_model_tooltip"))
        self.models_dir_label.setText(tr("local_models_dir_label"))
        self.models_dir_label.setToolTip(tr("local_models_dir_tooltip"))
        self.models_dir_button.setText(tr("ffmpeg_browse_button"))
        self.strategy_label.setText(tr("local_strategy_label"))
        self.strategy_label.setToolTip(tr("local_strategy_tooltip"))
        for index, strategy in enumerate(LOAD_STRATEGIES):
            self.strategy_selector.setItemText(index, tr(f"local_strategy_{strategy}"))
        self.idle_label.setText(tr("local_idle_label"))
        self.idle_label.setToolTip(tr("local_idle_tooltip"))
        self.idle_input.setSuffix(tr("local_idle_suffix"))
        self.idle_input.setSpecialValueText(tr("local_idle_never"))
        self.vad_checkbox.setText(tr("local_vad"))
        self.vad_checkbox.setToolTip(tr("local_vad_tooltip"))
        self.server_checkbox.setText(tr("local_server_enabled"))
        self.server_checkbox.setToolTip(tr("local_server_enabled_tooltip"))
        self.host_label.setText(tr("local_server_host_label"))
        self.port_label.setText(tr("local_server_port_label"))
        self.copy_endpoint_button.setText(tr("local_server_copy_button"))
        self.test_label.setText(tr("local_test_label"))
        self.test_label.setToolTip(tr("local_test_tooltip"))
        self.test_button.setText(tr("local_test_button"))
        self.bench_label.setText(tr("local_bench_label"))
        self.bench_label.setToolTip(tr("local_bench_tooltip"))
        self.bench_button.setText(tr("local_bench_stop" if self._bench is not None else "local_bench_button"))
        self.bench_file_button.setText(tr("local_bench_file_button"))
        self.bench_table.setHorizontalHeaderLabels([tr("local_bench_model"), tr("local_bench_load"),
                                                    tr("local_bench_first"), tr("local_bench_best"), "RTF"])
        self.log_label.setText(tr("local_log_label"))
        self.log_label.setToolTip(tr("local_log_tooltip"))
        self.log_button.setText(tr("local_log_button"))
        self.log_button.setToolTip(ENGINE_LOG_PATH)
        for row in self._rows:
            row.sync_texts()
        self._refresh_devices()
        self._update_enabled_state()


def status_text(translator: TranslationManager, status: EngineStatus, devices: List[Dict[str, str]]) -> str:
    """Translated one-line status of the engine (settings page and tray menu)."""
    tr = translator.tr
    device = next((tr(f"local_device_{d['id']}", name=d["name"]) for d in devices if d["id"] == status.device),
                  status.device)
    model = status.model.rsplit("/", 1)[-1]
    if status.state == READY:
        return tr("local_status_ready", device=device, model=model)
    if status.state == LOADING and status.downloading:
        total = status.download_total_bytes
        done = status.downloaded_bytes
        progress = (f"{_megabytes(done)} / {_megabytes(total)} ({min(100, 100 * done // total)} %)" if total
                    else _megabytes(done))
        return tr("local_status_downloading", model=model, progress=progress)
    if status.state == LOADING:
        return tr("local_status_loading", device=device, model=model)
    if status.state == STARTING:
        return tr("local_status_starting")
    if status.state == DOWN:
        return tr("local_status_down", detail=status.detail)
    if status.state == CONFLICT:
        return tr("local_status_conflict", detail=status.detail)
    if status.state == UNAVAILABLE:
        return tr("local_status_unavailable", reason=tr(status.detail) if status.detail else "")
    return tr("local_status_off_idle" if status.reason == "idle" else "local_status_off")
