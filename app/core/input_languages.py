"""Catalog and validation for the spoken languages offered in the recording palette."""

from __future__ import annotations

#: The stored value of "detect the language"; its caption is translated by whoever shows it.
DETECT_LANGUAGE = ""
#: The palette shows its languages as one-click pills on the microphone line; more would crowd it.
MAX_PALETTE_LANGUAGES = 5

# Map display names to ISO 639-1 codes, with automatic detection first.
LANGUAGES = {
    "Detect Language": DETECT_LANGUAGE,
    "English": "en",
    "German": "de",
    "French": "fr",
    "Spanish": "es",
    "Italian": "it",
    "Dutch": "nl",
    "Afrikaans": "af",
    "Arabic": "ar",
    "Armenian": "hy",
    "Azerbaijani": "az",
    "Belarusian": "be",
    "Bosnian": "bs",
    "Bulgarian": "bg",
    "Catalan": "ca",
    "Chinese": "zh",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Estonian": "et",
    "Finnish": "fi",
    "Galician": "gl",
    "Greek": "el",
    "Hebrew": "he",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Icelandic": "is",
    "Indonesian": "id",
    "Japanese": "ja",
    "Kannada": "kn",
    "Kazakh": "kk",
    "Korean": "ko",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Macedonian": "mk",
    "Malay": "ms",
    "Marathi": "mr",
    "Maori": "mi",
    "Nepali": "ne",
    "Norwegian": "no",
    "Persian": "fa",
    "Polish": "pl",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Serbian": "sr",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Swahili": "sw",
    "Swedish": "sv",
    "Tagalog": "tl",
    "Tamil": "ta",
    "Thai": "th",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Vietnamese": "vi",
    "Welsh": "cy",
}


def palette_languages(configured: object, current: str) -> list[str]:
    """The language codes the recording palette offers, in catalog order ("Auto" first).

    ``configured`` is the raw ``recording_palette_languages`` value; a hand-edited config may
    hold anything, so only known codes of a JSON list count. The ``current`` input language is
    always offered: at the limit it takes the last slot.
    """
    selected = {current}
    if isinstance(configured, list):
        selected.update(code for code in configured if isinstance(code, str))
    offered = [code for code in LANGUAGES.values() if code in selected]
    if current not in offered[MAX_PALETTE_LANGUAGES:]:
        return offered[:MAX_PALETTE_LANGUAGES]
    return offered[:MAX_PALETTE_LANGUAGES - 1] + [current]


def default_palette_languages(input_language: str) -> list[str]:
    """Offer automatic detection and the initial input language on a fresh config; the user adds the rest."""
    return palette_languages([DETECT_LANGUAGE], input_language)
