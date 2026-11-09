"""Search the settings window: which pages and settings match the typed words, and where they live.

Pure logic; the window collects the entries from its pages and opens the one the user picks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

#: Separates a page from the card a setting is on, such as "Transcription › Second pass".
PLACE_SEPARATOR = " › "


@dataclass(frozen=True, eq=False)
class SettingEntry:
    """A page, or a setting on it; ``target`` (the setting's widget) is None for the page itself.

    ``group`` is the caption of the card or section holding the setting ("" without one).
    """

    page_index: int
    page: str
    title: str
    description: str
    group: str = ""
    target: object = None


def find(query: str, entries: Sequence[SettingEntry]) -> List[SettingEntry]:
    """Entries containing every typed word, ignoring case; title matches come first.

    A word may also match the description, the page or the card, so "logging" finds the
    settings of the Logging card.
    """
    words = query.casefold().split()
    if not words:
        return []

    def has(text: str) -> bool:
        folded = text.casefold()
        return all(word in folded for word in words)

    titled = [entry for entry in entries if has(entry.title)]
    shown = {id(entry) for entry in titled}
    described = [entry for entry in entries if id(entry) not in shown
                 and has(f"{entry.title} {entry.description} {entry.page} {entry.group}")]
    return titled + described


def place(entry: SettingEntry, entries: Sequence[SettingEntry]) -> str:
    """Where ``entry`` lives: its page, and its card when another setting on that page has the same title."""
    title = entry.title.casefold()
    repeated = any(other is not entry and other.page_index == entry.page_index and other.target is not None
                   and other.title.casefold() == title for other in entries)
    if entry.target is None or not entry.group or not repeated:
        return entry.page
    return f"{entry.page}{PLACE_SEPARATOR}{entry.group}"
