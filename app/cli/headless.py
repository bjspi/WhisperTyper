"""Transcription and rephrasing without the tray app: synchronous, from the saved configuration.

The steps are the tray app's: the transcription request (``transcription_requests``), the
transcript check, the saved corrections and the rephrase routing (automatic prompt, LivePrompt
trigger words) — only the delivery differs: the caller receives the text instead of having it
typed. The configuration is read, migrated in memory and never written back.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from app.core.api_keys import (
    LOCAL_PROVIDER,
    GroqKeyRotation,
    provider_for_url,
    rephrasing_configured,
    selected_api_key,
    transcription_configured,
)
from app.core.config_store import ConfigStore
from app.core.constants import APP_DATA_DIR, CONFIG_FILE
from app.core.hotkeys import normalize_hotkey_string
from app.core.i18n import TranslationManager
from app.core.models import LOCAL_TRANSCRIPTION_MODEL
from app.core.prompts import INSTRUCTION, auto_apply_prompt, captioned_transformations
from app.core.rephrase_routing import RephrasePlan, plan_rephrase, usable_transcript
from app.core.replacements import ReplacementError, Replacements
from app.core.timing import OperationTiming
from app.local_engine.catalog import DEFAULT_SERVER_PORT, settings_from_config
from app.services.net import proxies_for
from app.services.rephrasing import RephrasingError, rephrase_text
from app.services.transcription import TranscriptionError, transcribe
from app.services.transcription_requests import TranscriptionTarget, build_request, configured_target, sent_prompt

#: Hears the progress phases of a transcription: "compressing", "transcribing", "rephrasing".
PhaseCallback = Callable[[str], None]
#: Input language value that lets the model detect the spoken language.
AUTO_LANGUAGE = "auto"
#: Seconds a transcription waits for the local engine of the running app to finish loading a model.
LOCAL_ENGINE_WAIT_S = 120.0
#: Device the engine settings are derived with; the address does not depend on it.
_ADDRESS_DEVICE = "cpu"
_LOCAL_ENGINE_UNREACHABLE = (
    "The local engine is reachable from the command line only while WhisperTyper runs with its "
    "server for other programs switched on (Settings → Local transcription)."
)


class HeadlessError(Exception):
    """A request cannot be made or failed; the message is meant for the user as-is."""


@dataclass(frozen=True)
class SavedPrompt:
    """A prompt of the Prompts settings page, numbered as listed there (from 1)."""

    number: int
    caption: str
    text: str
    is_instruction: bool
    auto_apply: bool


@dataclass(frozen=True)
class Transcript:
    """The outcome of one file: the delivered text and how it came about."""

    file: str
    text: str
    raw_text: str
    model: str
    language: str
    rephrased: bool
    seconds: float


@dataclass(frozen=True)
class TranscribeOptions:
    """Per-call overrides of the saved settings.

    ``system_prompt`` replaces the automatic rephrasing; ``raw`` skips corrections and every
    rephrasing, so the model's transcript is returned as it came.
    """

    language: Optional[str] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    raw: bool = False


class HeadlessSession:
    """The saved configuration, ready to transcribe and rephrase on the calling thread."""

    def __init__(self, config_file: str = CONFIG_FILE) -> None:
        """Load the configuration with its migrations applied in memory (the file stays untouched)."""
        self.config_file = config_file
        self.config: Dict[str, Any] = ConfigStore(config_file, normalize_hotkey_string).load(read_only=True)[0]
        self._translator = TranslationManager(self.config.get("ui_language", "en"))
        self._keys = GroqKeyRotation()
        self._replacements = _compile_replacements(self.config)

    # --- Prompts -------------------------------------------------------------------------
    def prompts(self) -> List[SavedPrompt]:
        """The usable prompts (caption and text set) in the order of the settings page."""
        return [SavedPrompt(number, entry["caption"].strip(), entry["text"].strip(), entry["kind"] == INSTRUCTION,
                            entry["auto_apply"])
                for number, entry in enumerate(captioned_transformations(self.config["post_rephrasing_entries"]), 1)]

    def find_prompt(self, name: str) -> SavedPrompt:
        """The prompt with this number, caption or unique caption start (case and emoji ignored)."""
        prompts = self.prompts()
        exact = [prompt for prompt in prompts if name.strip() in (str(prompt.number), prompt.caption)]
        if exact:
            return exact[0]
        # Emoji-only captions compare as "": they are found by their exact caption above only.
        wanted = _comparable(name)
        loose = [prompt for prompt in prompts if wanted and _comparable(prompt.caption) == wanted]
        if len(loose) == 1:
            return loose[0]
        candidates = [prompt for prompt in prompts if wanted and _comparable(prompt.caption).startswith(wanted)]
        if len(candidates) == 1:
            return candidates[0]
        captions = ", ".join(f"{prompt.number} {prompt.caption}" for prompt in candidates or prompts) or "none"
        problem = "is ambiguous" if candidates else "matches no saved prompt"
        raise HeadlessError(f'"{name}" {problem}. Prompts: {captions}')

    def auto_prompt(self) -> Optional[str]:
        """The prompt applied automatically to every transcription, when rephrasing can run."""
        if not rephrasing_configured(self.config):
            return None
        return auto_apply_prompt(self.config.get("post_rephrasing_entries", []))

    # --- Rephrasing ----------------------------------------------------------------------
    def rephrase(self, text: str, system_prompt: str, context: str = "") -> str:
        """Send ``text`` with ``system_prompt`` to the configured chat model and return its reply."""
        if not rephrasing_configured(self.config):
            raise HeadlessError("Rephrasing is not set up: add an API key in the settings (API keys page).")
        try:
            reply = rephrase_text(
                system_prompt=system_prompt,
                user_prompt=text,
                api_url=self.config["rephrasing_api_url"],
                api_key=selected_api_key(self.config, "rephrasing"),
                model=self.config["rephrasing_model"],
                temperature=self.config["rephrasing_temperature"],
                context=context,
                proxies=proxies_for(self.config),
            )
        except (RephrasingError, ValueError) as error:
            raise HeadlessError(str(error)) from error
        if not reply:
            raise HeadlessError("The rephrasing model returned an empty answer.")
        return reply

    # --- Transcription -------------------------------------------------------------------
    def transcribe(self, path: str, options: TranscribeOptions, on_phase: Optional[PhaseCallback] = None) -> Transcript:
        """Transcribe the audio or video file at ``path`` and post-process it like the tray app.

        ``on_phase`` hears "compressing" (extraction/compression with ffmpeg), "transcribing" and
        "rephrasing", for a progress display.
        """
        if not os.path.isfile(path):
            raise HeadlessError(f"File not found: {path}")
        config = self._config_for(options)
        if not transcription_configured(config):
            raise HeadlessError("Transcription is not set up: add an API key in the settings (API keys page).")
        notify = on_phase or _ignore_phase
        timing = OperationTiming("cli_transcription")
        started = time.monotonic()
        request = build_request(config, path, self._target(config), self._keys, timing.operation_id)
        try:
            raw_text = transcribe(request, timing, self._translator.tr,
                                  on_compressing=lambda _name: notify("compressing"),
                                  on_transcribing=lambda: notify("transcribing"))
        except TranscriptionError as error:
            raise HeadlessError(str(error)) from error
        text, rephrased = self._post_process(raw_text, config, options, notify)
        return Transcript(file=path, text=text, raw_text=raw_text, model=request.model,
                          language=request.language or AUTO_LANGUAGE, rephrased=rephrased,
                          seconds=round(time.monotonic() - started, 2))

    def _config_for(self, options: TranscribeOptions) -> Dict[str, Any]:
        """The saved configuration with this call's language and model."""
        config = dict(self.config)
        if options.language is not None:
            config["input_language"] = "" if options.language == AUTO_LANGUAGE else options.language
        if options.model:
            config["model"] = options.model
        return config

    def _target(self, config: Dict[str, Any]) -> TranscriptionTarget:
        """The configured provider; the local engine only through the running app's server."""
        target = configured_target(config)
        if provider_for_url(target.endpoint) != LOCAL_PROVIDER:
            return target
        if not (config.get("local_engine_enabled") and config.get("local_server_enabled")):
            return TranscriptionTarget(target.endpoint, LOCAL_TRANSCRIPTION_MODEL, local_error=_LOCAL_ENGINE_UNREACHABLE)
        port = int(config.get("local_server_port") or DEFAULT_SERVER_PORT)
        address = settings_from_config(config, _ADDRESS_DEVICE, APP_DATA_DIR, port).endpoint
        return TranscriptionTarget(address, LOCAL_TRANSCRIPTION_MODEL, local_wait_s=LOCAL_ENGINE_WAIT_S)

    def _post_process(self, raw_text: str, config: Dict[str, Any], options: TranscribeOptions,
                      notify: PhaseCallback) -> tuple[str, bool]:
        """The delivered text and whether a chat model rewrote it."""
        text = usable_transcript(raw_text, sent_prompt(config))
        if text is None:
            raise HeadlessError("No speech recognized.")
        if options.raw:
            return text, False
        if config["replacements_enabled"]:
            text = self._replacements.apply(text)[0]
        plan = self._plan(text, config, options)
        if plan is None:
            return text, False
        notify("rephrasing")
        return self.rephrase(plan.user_prompt, plan.system_prompt, plan.context), True

    def _plan(self, text: str, config: Dict[str, Any], options: TranscribeOptions) -> Optional[RephrasePlan]:
        """An explicit prompt wins; otherwise the app's routing (trigger words, automatic prompt)."""
        if options.system_prompt:
            return RephrasePlan(system_prompt=options.system_prompt, user_prompt=text)
        return plan_rephrase(text, config, None, "", self.auto_prompt())


def _ignore_phase(_phase: str) -> None:
    """Progress is not shown."""


def _compile_replacements(config: Dict[str, Any]) -> Replacements:
    """The saved correction rules; invalid rules disable corrections like in the app."""
    try:
        return Replacements(config["replacements_rules"])
    except ReplacementError as error:
        logging.warning("Correction rule %s is invalid; corrections are off until it is fixed in the settings.",
                        error.line)
        return Replacements("")


def _comparable(caption: str) -> str:
    """A caption reduced to lower-case letters, digits and single spaces (emoji and symbols dropped)."""
    kept = "".join(char if char.isalnum() else " " for char in caption.casefold())
    return " ".join(kept.split())
