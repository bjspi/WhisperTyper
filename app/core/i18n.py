"""Loads translated strings from JSON language files."""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

from app.core.paths import resource_path


class TranslationManager:
    """Manages loading and retrieving translated strings from JSON files."""

    def __init__(self, initial_language: str = 'en') -> None:
        """
        Initializes the TranslationManager.

        Args:
            initial_language (str): The initial language code (e.g., 'en').
        """
        self.translations: Dict[str, str] = {}
        self.language: str = ''
        self.set_language(initial_language)

    def get_base_path(self) -> str:
        """
        Gets the base path for resource files, compatible with PyInstaller.

        Returns:
            str: The absolute base path.
        """
        return resource_path()

    def set_language(self, lang_code: str) -> None:
        """
        Sets the current language and loads the corresponding translation file.

        Args:
            lang_code (str): The language code to load (e.g., 'en', 'de').
        """
        if self.language == lang_code:
            return

        base_path = self.get_base_path()
        translations_dir = os.path.join(base_path, 'lang')
        filepath = os.path.join(translations_dir, f"{lang_code}.json")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.translations = json.load(f)
            self.language = lang_code
            logging.info(f"Successfully loaded language: {lang_code}")
        except (FileNotFoundError, json.JSONDecodeError):
            logging.warning(f"Could not load language file for '{lang_code}'. Falling back to English.")
            # Fallback to English if the selected language file is missing/corrupt
            if lang_code != 'en':
                self.set_language('en')

    def tr(self, key: str, **kwargs: Any) -> str:
        """
        Retrieves a translated string for a given key.

        Args:
            key (str): The key for the string to translate.
            **kwargs: Placeholder values to format into the string.

        Returns:
            str: The translated and formatted string, or the key if not found.
        """
        text = self.translations.get(key, key)
        if kwargs:
            try:
                return text.format(**kwargs)
            except (KeyError, IndexError, ValueError, AttributeError):
                # A malformed placeholder in a translation file must never crash a caller
                # (these run inside signal slots); fall back to the raw string.
                return text
        return text
