"""Runs one rephrasing HTTP request off the GUI thread.

Self-contained by design: the caller snapshots all API settings on the GUI thread and
passes plain values in, so the worker never touches the live config dict (or any other
app state) from its thread.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

from PyQt6.QtCore import QObject, pyqtSignal

from app.services.rephrasing import rephrase_text


class RephrasingWorker(QObject):
    """Runs the rephrasing API request in a separate thread to avoid blocking the GUI."""

    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        api_url: str,
        api_key: str,
        model: str,
        temperature: float,
        context: str = "",
        proxies: Optional[Dict[str, str]] = None,
    ) -> None:
        """Store an immutable snapshot of everything the request needs.

        Args:
            system_prompt: The system-level instruction for the AI.
            user_prompt: The user's direct input or text to be processed.
            api_url: Chat-completions endpoint URL.
            api_key: Bearer token for the endpoint.
            model: Model identifier.
            temperature: Sampling temperature.
            context: Additional context (e.g. selected text).
            proxies: Optional ``requests`` proxies mapping (resolved by the caller).
        """
        super().__init__()
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.context = context
        self.proxies = proxies

    def run(self) -> None:
        """Execute the rephrasing request and emit ``finished`` or ``error``."""
        logging.info("RephrasingWorker started.")
        try:
            rephrased_text = rephrase_text(
                system_prompt=self.system_prompt,
                user_prompt=self.user_prompt,
                api_url=self.api_url,
                api_key=self.api_key,
                model=self.model,
                temperature=self.temperature,
                context=self.context,
                proxies=self.proxies,
            )
        except Exception as e:
            error_msg = f"An unexpected error occurred in RephrasingWorker:\n{e}"
            logging.error(error_msg)
            self.error.emit(error_msg)
            return

        if rephrased_text:
            self.finished.emit(rephrased_text)
        else:
            self.error.emit("Rephrasing resulted in empty text.")
