"""Prompt captions with emoji: compact palette labels and a lossless config round trip."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("PyQt6.QtCore")

from app.core.config_store import ConfigStore  # noqa: E402
from app.core.hotkeys import normalize_hotkey_string  # noqa: E402
from app.ui.floating_buttons import compact_caption  # noqa: E402


@pytest.mark.parametrize("caption, expected", [
    ("\U0001F468\u200d\U0001F4BB Code", "\U0001F468\u200d\U0001F4BB"),  # ZWJ sequence stays whole
    ("\U0001F44D\U0001F3FD", "\U0001F44D\U0001F3FD"),                    # skin tone modifier
    ("\U0001F1E9\U0001F1EA Deutsch", "\U0001F1E9\U0001F1EA"),            # flag (regional indicators)
    ("Polish", "Pol"),
    ("Hi \U0001F44B", "Hi"),
])
def test_compact_label_never_splits_a_character(caption, expected):
    assert compact_caption(caption) == expected


def test_emoji_captions_survive_save_and_load_unchanged(tmp_path):
    caption = "\U0001F468\u200d\U0001F4BB Code \U0001F1E9\U0001F1EA Übersetzen"
    path = tmp_path / "config.json"
    store = ConfigStore(str(path), normalize_hotkey_string)
    config, _changed = store.load()
    config["post_rephrasing_entries"] = [{"caption": caption, "text": "Rewrite \U0001F44D\U0001F3FD",
                                          "show_during_recording": True}]
    store.save(config)
    assert caption in path.read_text(encoding="utf-8")  # stored as UTF-8, not \u escapes
    reloaded, _changed = ConfigStore(str(path), normalize_hotkey_string).load()
    entry = reloaded["post_rephrasing_entries"][1]  # after the instruction entry the migration adds
    assert (entry["caption"], entry["text"]) == (caption, "Rewrite \U0001F44D\U0001F3FD")
    assert json.loads(path.read_text(encoding="utf-8"))["post_rephrasing_entries"][0]["caption"] == caption
