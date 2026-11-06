"""The command line: grouped help, prompt lookup, output modes and the headless pipeline."""
from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

import app.cli.headless as headless_module
import app.cli.main as main_module
from app.cli.headless import HeadlessError, HeadlessSession, TranscribeOptions, Transcript
from app.cli.main import cli

OPENAI_KEY = {"id": "o", "name": "OpenAI", "provider": "openai", "key": "sk-test-0000000000000000"}
ENTRIES = [
    {"kind": "instruction", "caption": "✨", "text": "Carry out the order.", "enabled": True},
    {"caption": "📧 Nice mail", "text": "Write a friendly mail.", "auto_apply": True},
    {"caption": "Translate EN", "text": "Translate to English."},
]


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"api_key_profiles": [OPENAI_KEY], "post_rephrasing_entries": ENTRIES,
                                "replacements_enabled": True, "replacements_rules": "Turbotype ; TurboType ; 1"}),
                    encoding="utf-8")
    return str(path)


@pytest.fixture
def session(config_file):
    return HeadlessSession(config_file)


@pytest.fixture
def runner(monkeypatch, session):
    monkeypatch.setattr(main_module, "HeadlessSession", lambda: session)
    return CliRunner()


def test_help_shows_the_command_groups(runner):
    result = runner.invoke(cli, ["--help"], prog_name=main_module.PROGRAM)
    assert result.exit_code == 0
    assert "Speech" in result.output and "Application" in result.output
    assert "transcribe" in result.output and "rephrase" in result.output


@pytest.mark.parametrize(("problems", "exit_code"), [([], 0), (["no access"], main_module.EXIT_FAILED)],
                         ids=["success", "failure"])
def test_install_integration_follows_the_settings_and_reports_problems(runner, session, monkeypatch, problems, exit_code):
    system_integration = pytest.importorskip("app.controllers.system_integration")
    seen = []
    monkeypatch.setattr(system_integration, "apply_now", lambda config: seen.append(config) or problems)
    result = runner.invoke(cli, ["install-integration"])
    assert result.exit_code == exit_code
    assert seen == [session.config]


def test_loading_never_writes_the_config_file(config_file):
    before = open(config_file, encoding="utf-8").read()
    HeadlessSession(config_file)
    assert open(config_file, encoding="utf-8").read() == before


@pytest.mark.parametrize(("name", "caption"), [
    ("2", "📧 Nice mail"),
    ("nice mail", "📧 Nice mail"),
    ("translate", "Translate EN"),
    ("✨", "✨"),
])
def test_prompts_are_found_by_number_caption_or_unique_start(session, name, caption):
    assert session.find_prompt(name).caption == caption


def test_an_unknown_prompt_names_the_choices(session):
    with pytest.raises(HeadlessError, match="Nice mail"):
        session.find_prompt("poem")


def test_prompts_json_lists_number_caption_and_automatic_flag(runner):
    result = runner.invoke(cli, ["prompts", "--json"])
    listed = json.loads(result.output)
    assert [(item["number"], item["caption"], item["auto_apply"]) for item in listed] == [
        (1, "✨", False), (2, "📧 Nice mail", True), (3, "Translate EN", False)]


def test_rephrase_reads_stdin_and_uses_the_automatic_prompt(runner, monkeypatch, session):
    calls = []
    monkeypatch.setattr(session, "rephrase", lambda text, prompt, context: calls.append((text, prompt)) or "Done.")
    result = runner.invoke(cli, ["rephrase", "--json"], input="hi there\n")
    assert result.exit_code == 0
    assert json.loads(result.output) == {"text": "Done.", "prompt": "📧 Nice mail"}
    assert calls == [("hi there", "Write a friendly mail.")]


def test_redirected_output_is_the_plain_text(runner, monkeypatch, session):
    monkeypatch.setattr(session, "rephrase", lambda text, prompt, context: "Plain result")
    result = runner.invoke(cli, ["rephrase", "x", "-s", "Do it"])
    assert result.output == "Plain result\n"


def test_prompt_and_raw_exclude_each_other(runner, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    result = runner.invoke(cli, ["transcribe", str(audio), "--raw", "-p", "2"])
    assert result.exit_code == 2


def test_a_failed_file_is_reported_and_the_others_still_run(runner, monkeypatch, session, tmp_path):
    good, bad = tmp_path / "good.wav", tmp_path / "bad.wav"
    good.write_bytes(b"RIFF")
    bad.write_bytes(b"RIFF")

    def transcribe(path, options, on_phase=None):
        if path.endswith("bad.wav"):
            raise HeadlessError("No speech recognized.")
        return Transcript(path, "Hello.", " Hello.", "whisper-1", "en", False, 0.5)

    monkeypatch.setattr(session, "transcribe", transcribe)
    result = runner.invoke(cli, ["transcribe", str(bad), str(good), "--json"])
    assert result.exit_code == 1
    outcomes = json.loads(result.stdout)
    assert [(item["ok"], item.get("text")) for item in outcomes] == [(False, None), (True, "Hello.")]


def test_transcription_applies_corrections_then_the_automatic_prompt(monkeypatch, session, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    session.config["api_key_profiles"] = [{**OPENAI_KEY, "id": "t"}]
    monkeypatch.setattr(headless_module, "transcription_configured", lambda config: True)
    monkeypatch.setattr(headless_module, "transcribe", lambda request, timing, tr, **_kw: " Turbotype rocks")
    sent = []
    monkeypatch.setattr(session, "rephrase", lambda text, prompt, context="": sent.append((text, prompt)) or "Mail")
    result = session.transcribe(str(audio), TranscribeOptions(language="auto"))
    assert (result.text, result.raw_text, result.language, result.rephrased) == ("Mail", " Turbotype rocks", "auto", True)
    assert sent == [("TurboType rocks", "Write a friendly mail.")]


def test_raw_transcription_skips_corrections_and_rephrasing(monkeypatch, session, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setattr(headless_module, "transcription_configured", lambda config: True)
    monkeypatch.setattr(headless_module, "transcribe", lambda request, timing, tr, **_kw: "Turbotype rocks")
    result = session.transcribe(str(audio), TranscribeOptions(raw=True))
    assert (result.text, result.rephrased) == ("Turbotype rocks", False)


def test_local_engine_without_the_app_server_is_a_clear_error(session):
    session.config.update(api_endpoint="local://engine/v1/audio/transcriptions", local_engine_enabled=True,
                          local_server_enabled=False)
    target = session._target(session.config)
    assert "server for other programs" in target.local_error


@pytest.mark.parametrize(("args", "expected"), [
    ([], (False, None)),
    (["-r", "--show-settings", "3"], (True, 3)),  # what a restart from the tray passes
    (["run", "--show-settings"], (False, 0)),
])
def test_without_a_command_the_tray_app_starts(runner, monkeypatch, args, expected):
    started = []
    monkeypatch.setattr(main_module, "_run_tray", lambda restart, tab, foreground: started.append((restart, tab)) or 0)
    result = runner.invoke(cli, args)
    assert result.exit_code == 0
    assert started == [expected]
