"""The ``whispertyper`` command: grouped commands with coloured help and output.

Without a command it starts the tray app (as before). The speech commands use the saved
settings — keys, models, prompts, corrections — and print the result: decorated for people,
plain when redirected, and as JSON with ``--json`` for other tools.
"""
from __future__ import annotations

import dataclasses
import logging
import platform
import sys
from typing import Any, Dict, List, NoReturn, Optional

import rich_click as click
from rich.logging import RichHandler
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from app import __version__
from app.cli import output
from app.cli.headless import AUTO_LANGUAGE, HeadlessError, HeadlessSession, SavedPrompt, TranscribeOptions, Transcript
from app.core.api_keys import PROVIDER_NAMES, masked_api_key, provider_for_url, selected_api_key
from app.core.constants import APP_DATA_DIR, LANGUAGES
from app.services.gitutil import find_git_root, project_root

#: Exit code of a failed request; success is 0, and click exits with 2 on usage errors.
EXIT_FAILED = 1
#: Characters of a prompt text shown in the ``prompts`` table.
PROMPT_PREVIEW_CHARS = 70
#: Help look: rich-click theme (colour scheme and layout).
HELP_THEME = "cyan2-modern"
#: The command that hosts the tray app; every other command logs to stderr instead of the app's log file.
_TRAY_COMMAND = "run"
_PHASE_MESSAGES = {
    "compressing": "Extracting or compressing the audio …",
    "transcribing": "Transcribing …",
    "rephrasing": "Rephrasing …",
}
_LANGUAGE_CODES = sorted({code for code in LANGUAGES.values() if code} | {AUTO_LANGUAGE})

json_option = click.option("--json", "as_json", is_flag=True, help="Print one JSON document (for scripts and tools).")


@click.group(invoke_without_command=True, context_settings={"help_option_names": ["-h", "--help"]})
@click.rich_config(help_config=click.RichHelpConfiguration(theme=HELP_THEME, text_markup="rich"))
@click.command_panel("Speech", commands=["transcribe", "rephrase", "prompts"],
                     help="Use the saved keys, models and prompts — no running app needed.")
@click.command_panel("Application", commands=["run", "info"])
@click.option("-v", "--verbose", is_flag=True, help="Log requests and timings on stderr.")
@click.option("-r", "--restart", is_flag=True, hidden=True)
@click.option("--show-settings", type=int, is_flag=False, flag_value=0, default=None, hidden=True)
@click.version_option(__version__, "-V", "--version", prog_name="WhisperTyper", message="%(prog)s %(version)s")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, restart: bool, show_settings: Optional[int]) -> None:
    """[bold cyan]WhisperTyper[/] — voice to text and live AI prompting.

    Without a command, WhisperTyper starts in the system tray.
    """
    if ctx.invoked_subcommand is None:
        ctx.exit(_run_tray(restart, show_settings, foreground=False))
    if ctx.invoked_subcommand != _TRAY_COMMAND:
        _configure_logging(verbose)


# --- Application ---------------------------------------------------------------------------
@cli.command()
@click.option("-r", "--restart", is_flag=True, help="Stop a running WhisperTyper first and start afresh.")
@click.option("--show-settings", "show_settings", type=int, is_flag=False, flag_value=0, default=None,
              metavar="TAB", help="Open the settings window on this tab after startup.")
@click.option("--foreground", is_flag=True, help="Keep the app attached to this terminal and log here.")
def run(restart: bool, show_settings: Optional[int], foreground: bool) -> None:
    """Start WhisperTyper in the system tray (the default without a command).

    Started in a terminal, the app detaches from it, so the terminal stays free.
    """
    sys.exit(_run_tray(restart, show_settings, foreground))


@cli.command()
@json_option
def info(as_json: bool) -> None:
    """Show version, folders and the configured providers and models."""
    session = _session()
    report = _info_report(session)
    if as_json:
        output.print_json(report)
        return
    output.stdout.print(f"[accent]WhisperTyper[/accent] [bold]{__version__}[/bold]\n")
    for section, rows in report.items():
        if isinstance(rows, dict):
            output.facts_table(section.capitalize(), [(_label(key), _shown(value)) for key, value in rows.items()])


# --- Speech --------------------------------------------------------------------------------
@cli.command()
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False))
@click.option_panel("Transcription", options=["--language", "--model"])
@click.option_panel("Rephrasing", options=["--prompt", "--system", "--raw"],
                    help="By default the result is post-processed like in the app.")
@click.option_panel("Output", options=["--json", "--help"])
@click.option("-l", "--language", type=click.Choice(_LANGUAGE_CODES, case_sensitive=False), metavar="CODE",
              help="Spoken language (e.g. de, en), or 'auto' to detect it. Default: from the settings.")
@click.option("-m", "--model", metavar="MODEL", help="Transcription model of the configured provider.")
@click.option("-p", "--prompt", "prompt_name", metavar="NAME",
              help="Rephrase with a saved prompt: its number or caption (see 'prompts').")
@click.option("-s", "--system", "system_prompt", metavar="TEXT", help="Rephrase with this instruction instead.")
@click.option("--raw", is_flag=True, help="The model's transcript as it is: no corrections, no rephrasing.")
@json_option
def transcribe(files: tuple[str, ...], language: Optional[str], model: Optional[str], prompt_name: Optional[str],
               system_prompt: Optional[str], raw: bool, as_json: bool) -> None:
    """Transcribe audio or video files.

    Corrections, the automatic prompt and LivePrompt trigger words apply as in the app,
    unless [bold]--raw[/] is given. Several files are transcribed one after the other.

    [dim]Example:[/] whispertyper transcribe memo.m4a -p "Email" --json
    """
    _exclusive(prompt=prompt_name, system=system_prompt, raw=raw)
    session = _session()
    instruction = _prompt_text(session, prompt_name) if prompt_name else system_prompt
    options = TranscribeOptions(language=language.lower() if language else None, model=model,
                                system_prompt=instruction, raw=raw)
    outcomes = [_transcribe_one(session, path, options) for path in files]
    if as_json:
        output.print_json([_transcript_json(path, outcome) for path, outcome in zip(files, outcomes)])
    else:
        _show_transcripts(outcomes)
    if not all(isinstance(outcome, Transcript) for outcome in outcomes):
        sys.exit(EXIT_FAILED)


@cli.command()
@click.argument("text", required=False)
@click.option("-p", "--prompt", "prompt_name", metavar="NAME",
              help="A saved prompt: its number or caption (see 'prompts'). Default: the automatic prompt.")
@click.option("-s", "--system", "system_prompt", metavar="TEXT", help="Use this instruction instead of a saved prompt.")
@click.option("-c", "--context", default="", metavar="TEXT", help="Extra context the model may use.")
@json_option
def rephrase(text: Optional[str], prompt_name: Optional[str], system_prompt: Optional[str], context: str,
             as_json: bool) -> None:
    """Rewrite TEXT with a prompt (TEXT '-' or left out: read from stdin).

    [dim]Example:[/] echo "hi, can u send it tmrw" | whispertyper rephrase -p "Formal"
    """
    _exclusive(prompt=prompt_name, system=system_prompt)
    session = _session()
    source = _input_text(text)
    caption, instruction = _rephrase_instruction(session, prompt_name, system_prompt)
    try:
        with output.stderr.status("[accent]Rephrasing …[/accent]"):
            reply = session.rephrase(source, instruction, context)
    except HeadlessError as error:
        _fail(str(error))
    if as_json:
        output.print_json({"text": reply, "prompt": caption})
    elif output.decorated():
        output.result_panel(reply, caption or "Rephrased", session.config["rephrasing_model"])
    else:
        output.print_plain(reply)


@cli.command()
@json_option
def prompts(as_json: bool) -> None:
    """List the saved prompts that 'transcribe' and 'rephrase' can use."""
    saved = _session().prompts()
    if as_json:
        output.print_json(saved)
        return
    if not saved:
        output.warning("No prompts saved yet (Settings → Prompts).")
        return
    _show_prompts(saved)


# --- Helpers -------------------------------------------------------------------------------
def _run_tray(restart: bool, show_settings: Optional[int], foreground: bool) -> int:
    """Start the tray app — in a detached process when started in a terminal; Qt is imported only then."""
    from app import host

    if foreground or not host.started_from_terminal():
        return host.run_tray(restart, show_settings)
    arguments = ["run", *(["--restart"] if restart else [])]
    if show_settings is not None:
        arguments += ["--show-settings", str(show_settings)]
    host.relaunch_detached(arguments)
    output.success("WhisperTyper runs in the system tray.")
    return 0


def _configure_logging(verbose: bool) -> None:
    """Warnings on stderr; ``--verbose`` adds the requests and timings."""
    handler = RichHandler(console=output.stderr, show_path=False, markup=False, rich_tracebacks=False)
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING, format="%(message)s",
                        handlers=[handler], force=True)


def _session() -> HeadlessSession:
    """The saved settings; an unreadable configuration ends the command."""
    try:
        return HeadlessSession()
    except (KeyError, TypeError, ValueError) as error:
        _fail(f"The configuration cannot be used: {error}")


def _fail(message: str, hint: Optional[str] = None) -> NoReturn:
    """Report ``message`` and end the command unsuccessfully."""
    output.error(message, hint)
    sys.exit(EXIT_FAILED)


def _exclusive(**choices: Any) -> None:
    """At most one of the given options may be set."""
    given = [f"--{name}" for name, value in choices.items() if value]
    if len(given) > 1:
        raise click.UsageError(f"Use only one of {', '.join(given)}.")


def _prompt_text(session: HeadlessSession, name: str) -> str:
    """The text of the saved prompt ``name``; an unknown name ends the command."""
    try:
        return session.find_prompt(name).text
    except HeadlessError as error:
        _fail(str(error), "List them with: whispertyper prompts")


def _rephrase_instruction(session: HeadlessSession, prompt_name: Optional[str],
                          system_prompt: Optional[str]) -> tuple[Optional[str], str]:
    """Caption (None for an ad-hoc instruction) and system prompt of a rephrase call."""
    if system_prompt:
        return None, system_prompt
    if prompt_name:
        try:
            chosen = session.find_prompt(prompt_name)
        except HeadlessError as error:
            _fail(str(error), "List them with: whispertyper prompts")
        return chosen.caption, chosen.text
    automatic = next((prompt for prompt in session.prompts() if prompt.auto_apply), None)
    if automatic is None:
        _fail("No prompt chosen and none is applied automatically.", "Pass --prompt NAME or --system TEXT.")
    return automatic.caption, automatic.text


def _input_text(text: Optional[str]) -> str:
    """TEXT, or stdin for '-' or no argument; empty input ends the command."""
    if text is not None and text != "-":
        source = text
    elif sys.stdin is None or sys.stdin.isatty():
        raise click.UsageError("Give TEXT or pipe it in on stdin.")
    else:
        source = _read_stdin()
    if not source.strip():
        _fail("The text is empty.")
    return source.strip()


def _read_stdin() -> str:
    """Piped text, read as UTF-8 (falling back to the system encoding for other pipes)."""
    data = sys.stdin.buffer.read()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode(sys.stdin.encoding or "utf-8", errors="replace")


def _transcribe_one(session: HeadlessSession, path: str, options: TranscribeOptions) -> Transcript | HeadlessError:
    """One file's transcript, or the error that stopped it (reported at once, the next file follows)."""
    def progress(phase: str) -> str:
        return f"[accent]{_PHASE_MESSAGES[phase]}[/accent] [muted]{escape(path)}[/muted]"

    status = output.stderr.status(progress("transcribing"))
    with status:
        try:
            return session.transcribe(path, options, on_phase=lambda phase: status.update(progress(phase)))
        except HeadlessError as error:
            output.error(f"{path}: {error}")
            return error


def _show_transcripts(outcomes: List[Transcript | HeadlessError]) -> None:
    """Panels in a terminal; otherwise the plain texts, separated by a blank line."""
    transcripts = [outcome for outcome in outcomes if isinstance(outcome, Transcript)]
    if not output.decorated():
        output.print_plain("\n\n".join(transcript.text for transcript in transcripts))
        return
    for transcript in transcripts:
        details = [transcript.model, transcript.language, f"{transcript.seconds:.1f} s"]
        if transcript.rephrased:
            details.append("rephrased")
        output.result_panel(transcript.text, transcript.file, " · ".join(details))


def _transcript_json(path: str, outcome: Transcript | HeadlessError) -> Dict[str, Any]:
    """One file in the JSON result: the transcript fields, ``ok`` and ``error``."""
    if isinstance(outcome, Transcript):
        return {**dataclasses.asdict(outcome), "ok": True, "error": None}
    return {"file": path, "ok": False, "error": str(outcome)}


def _show_prompts(saved: List[SavedPrompt]) -> None:
    """The prompts as a table: number, caption, kind and the start of the text."""
    table = Table(header_style="accent", border_style="cyan", show_lines=False)
    table.add_column("#", justify="right", style="bold")
    table.add_column("Caption", style="bold")
    table.add_column("Kind")
    table.add_column("Text", style="muted", overflow="ellipsis", no_wrap=True)
    for prompt in saved:
        kind = "instruction" if prompt.is_instruction else "prompt"
        if prompt.auto_apply:
            kind += " [ok](automatic)[/ok]"
        preview = " ".join(prompt.text.split())
        if len(preview) > PROMPT_PREVIEW_CHARS:
            preview = preview[:PROMPT_PREVIEW_CHARS - 1] + "…"
        table.add_row(str(prompt.number), Text(prompt.caption), kind, Text(preview))
    output.stdout.print(table)


def _info_report(session: HeadlessSession) -> Dict[str, Any]:
    """Version, folders and providers; keys appear masked."""
    config = session.config
    checkout = find_git_root()
    automatic = next((prompt.caption for prompt in session.prompts() if prompt.auto_apply), None)
    return {
        "version": __version__,
        "installation": {
            "kind": "git checkout" if checkout else "package",
            "folder": checkout or project_root(),
            "data_folder": APP_DATA_DIR,
            "config_file": session.config_file,
            "python": platform.python_version(),
        },
        "transcription": {
            "provider": _provider(config["api_endpoint"]),
            "model": config["model"],
            "language": config["input_language"] or AUTO_LANGUAGE,
            "api_key": _masked(selected_api_key(config, "transcription")),
        },
        "rephrasing": {
            "provider": _provider(config["rephrasing_api_url"]),
            "model": config["rephrasing_model"],
            "api_key": _masked(selected_api_key(config, "rephrasing")),
            "automatic_prompt": automatic,
        },
    }


def _provider(url: str) -> str:
    """Display name of the provider behind ``url``."""
    provider = provider_for_url(url)
    return PROVIDER_NAMES.get(provider, provider.capitalize())


def _masked(key: str) -> Optional[str]:
    """The key with its middle hidden; None without a key."""
    return masked_api_key(key) if key else None


def _label(key: str) -> str:
    """A report key as a label: ``data_folder`` → ``Data folder``."""
    return key.replace("_", " ").capitalize()


def _shown(value: Any) -> str:
    """A report value for people: None as a dash."""
    return "—" if value is None else str(value)


def main() -> None:
    """Entry point of the ``whispertyper`` command and of ``run.py``."""
    _prefer_utf8_output()
    cli(prog_name="whispertyper")


def _prefer_utf8_output() -> None:
    """Redirected output is UTF-8, whatever the console code page (umlauts, emoji in captions)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None and not stream.isatty():
            reconfigure(encoding="utf-8")
