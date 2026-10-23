"""Expert settings: the config.json-only "expert" block, its validation and its use when pasting."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

import app.controllers.text_output as text_output_module
from app.controllers.text_output import TextOutput
from app.core.config_store import ConfigStore
from app.core.expert_settings import EXPERT_SETTINGS, expert_defaults, expert_value, normalize_expert
from app.core.hotkeys import normalize_hotkey_string
from app.core.timing import OperationTiming


def test_defaults_are_the_previous_fixed_timings():
    # Introducing the block must not change any behaviour.
    assert expert_defaults() == {
        "paste_prepare_ms": 100, "paste_settle_ms": 100, "clipboard_restore_ms": 500,
        "clipboard_restore_fast_ms": 600, "copy_settle_ms": 100, "select_all_settle_ms": 80,
    }


@pytest.mark.parametrize(("raw", "expected", "warns"), [
    (250, 250, False),
    (250.0, 250, False),
    ("abc", 100, True),
    ("250", 100, True),
    (True, 100, True),
    (None, 100, True),
    (12.5, 100, True),
    (-5, 0, True),
    (99999, 2000, True),
])
def test_invalid_values_fall_back_to_the_default_and_out_of_range_ones_are_clamped(raw, expected, warns):
    block, warnings = normalize_expert({**expert_defaults(), "paste_prepare_ms": raw})
    assert block["paste_prepare_ms"] == expected
    assert bool(warnings) is warns
    assert all("paste_prepare_ms" in warning for warning in warnings)


def test_missing_keys_are_filled_and_unknown_keys_kept_with_a_warning():
    block, warnings = normalize_expert({"paste_settle_ms": 30, "paste_settel_ms": 40})
    assert block == {**expert_defaults(), "paste_settle_ms": 30, "paste_settel_ms": 40}
    assert warnings == ['Expert settings: unknown key "paste_settel_ms" is ignored.']


@pytest.mark.parametrize("raw", ["abc", [1, 2], 5, None])
def test_a_block_that_is_no_object_is_replaced_by_the_defaults(raw):
    block, warnings = normalize_expert(raw)
    assert block == expert_defaults()
    assert len(warnings) == 1


def test_expert_value_falls_back_without_a_valid_block():
    assert expert_value({}, "copy_settle_ms") == 100
    assert expert_value({"expert": "abc"}, "copy_settle_ms") == 100
    assert expert_value({"expert": {"copy_settle_ms": "x"}}, "copy_settle_ms") == 100
    assert expert_value({"expert": {"copy_settle_ms": 5000}}, "copy_settle_ms") == EXPERT_SETTINGS["copy_settle_ms"].maximum


def _load(tmp_path: Path, data: dict) -> tuple[ConfigStore, dict, bool]:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    store = ConfigStore(str(path), normalize_hotkey_string)
    config, changed = store.load()
    return store, config, changed


def test_load_adds_the_block_with_defaults(tmp_path):
    store, config, changed = _load(tmp_path, {})
    assert config["expert"] == expert_defaults()
    assert changed and store.load_warnings == []


def test_load_repairs_nonsense_and_reports_it(tmp_path):
    store, config, changed = _load(tmp_path, {"expert": {"paste_prepare_ms": "abc", "copy_settle_ms": 20}})
    assert config["expert"] == {**expert_defaults(), "copy_settle_ms": 20}
    assert changed
    assert len(store.load_warnings) == 1 and "paste_prepare_ms='abc'" in store.load_warnings[0]


def test_a_valid_block_loads_unchanged(tmp_path):
    custom = {**expert_defaults(), "paste_prepare_ms": 30}
    store, _config, _changed = _load(tmp_path, {"expert": custom})
    store.save(_config)
    store, config, _changed = _load(tmp_path, json.loads(Path(store.config_file).read_text(encoding="utf-8")))
    assert config["expert"] == custom and store.load_warnings == []


def test_paste_and_copy_use_the_configured_timings(monkeypatch):
    config = {"restore_clipboard": True, "fast_paste": False,
              "expert": {**expert_defaults(), "paste_prepare_ms": 30, "paste_settle_ms": 40,
                         "clipboard_restore_ms": 700, "copy_settle_ms": 50, "select_all_settle_ms": 60}}
    output = TextOutput(config, lambda key, **_kwargs: key, Mock(), lambda _permission: None)
    monkeypatch.setattr(output, "_simulate_key_combination", lambda _char: True)
    monkeypatch.setattr(text_output_module.ClipboardSnapshot, "capture", classmethod(lambda cls: cls("old")))
    monkeypatch.setattr(text_output_module.copykitten, "copy", Mock())
    monkeypatch.setattr(text_output_module.copykitten, "paste", Mock(return_value="selected"))
    restore = Mock()
    monkeypatch.setattr(output, "_schedule_clipboard_restore", restore)
    sleeps = Mock()
    monkeypatch.setattr(text_output_module.time, "sleep", sleeps)

    assert output.insert("text", OperationTiming("recording"))
    assert [call.args[0] for call in sleeps.call_args_list] == [0.03, 0.04]
    assert restore.call_args.args[1] == 700

    sleeps.reset_mock()
    monkeypatch.setattr(text_output_module.ClipboardSnapshot, "restore", lambda self: None)
    assert output.get_selected_text(select_all_first=True) == "selected"
    assert [call.args[0] for call in sleeps.call_args_list] == [0.06, 0.05]
