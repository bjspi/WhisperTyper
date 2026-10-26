"""Install the optional packages of the local engine with pip, showing pip's output live."""
from __future__ import annotations

import logging
import sys
from typing import List

from PyQt6.QtWidgets import QWidget

from app.core.i18n import TranslationManager
from app.platform import installation
from app.ui import message_dialog
from app.ui.package_install import run_install


def environment_description() -> str:
    """Where pip installs to: a virtual environment or the Python installation itself."""
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    return f"{'venv' if in_venv else 'Python'}: {sys.prefix}"


def confirm_install(parent: QWidget, translator: TranslationManager, requirements: List[str]) -> bool:
    """Ask before anything is downloaded; names the packages, the size and the target environment."""
    tr = translator.tr
    nvidia = any(req.startswith("nvidia-") for req in requirements)
    text = tr("local_install_confirm_text", packages="\n".join(f"• {req}" for req in requirements),
              size=tr("local_install_size_cuda" if nvidia else "local_install_size"),
              environment=environment_description())
    return message_dialog.ask(parent, tr("local_install_title"), text, tr("install_button"),
                              tr("dialog_cancel_button"), message_dialog.UPDATE)


def install(parent: QWidget, translator: TranslationManager, requirements: List[str]) -> bool:
    """Confirm, then install ``requirements``; True when they are installed afterwards."""
    if not requirements:
        return True
    if not confirm_install(parent, translator, requirements):
        logging.info("local_engine_install declined by the user: %s", requirements)
        return False
    command = installation.install_command(requirements)
    if command is None:
        logging.error("local_engine_install: neither pip nor uv is available for %s", sys.executable)
        return False
    return run_install(parent, translator.tr, translator.tr("local_install_title"), command)
