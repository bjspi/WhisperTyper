"""Build a transcription request from the saved configuration — no Qt, shared by the app and the CLI.

The tray app and the command line send the same request for the same settings: credential
rotation, hedging, the quality fallback and the ffmpeg lookup are decided here once. Where the
request goes (the configured provider, or the local engine's real address) is the caller's
``TranscriptionTarget``, because only the caller knows whether the local engine is running.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from app.core.api_keys import GroqKeyRotation, is_local_endpoint, provider_for_url
from app.core.models import transcription_reports_confidence
from app.core.prompts import transcription_prompt_for
from app.services.ffmpeg import resolve_ffmpeg
from app.services.net import proxies_for
from app.services.transcription import DEFAULT_HEDGE_RESPONSE_WAIT_S, TranscriptionRequest


@dataclass(frozen=True)
class TranscriptionTarget:
    """Where a transcription is sent: endpoint URL and model, plus the local engine's state.

    ``local_wait_s`` lets a request wait for a starting local engine; ``local_error`` is the
    translated reason why the local engine cannot transcribe (the request then fails at once).
    """

    endpoint: str
    model: str
    local_wait_s: float = 0.0
    local_error: str = ""


def configured_target(config: Mapping[str, Any]) -> TranscriptionTarget:
    """The provider endpoint and model as saved in the settings."""
    return TranscriptionTarget(endpoint=config["api_endpoint"], model=config["model"])


def sent_prompt(config: Mapping[str, Any]) -> str:
    """The transcription prompt for the configured input language (see ``transcription_prompt_for``)."""
    return transcription_prompt_for(config["prompt"], config["input_language"])


def build_request(config: Mapping[str, Any], audio_path: str, target: TranscriptionTarget,
                  keys: GroqKeyRotation, operation_id: str,
                  recording_format: Optional[str] = None) -> TranscriptionRequest:
    """Snapshot ``config`` into the request for ``audio_path``.

    ``recording_format`` is set for retained microphone recordings; the worker then resolves
    ffmpeg only if a long recording exceeds the upload limit. Picked files resolve it now, for
    video extraction and oversized-file compression.
    """
    endpoint, model = target.endpoint, target.model
    provider = provider_for_url(endpoint)
    api_key = keys.next_key(config)
    logging.info("transcription_credential op=%s provider=%s profile_id=%s rotation=%s",
                 operation_id, provider, keys.last_profile_id,
                 bool(config["groq_key_rotation"] and provider == "groq"))
    # Hedging: a stalled request is raced by a second one with the next key (rotation) or the same key.
    # A server on this computer has no network path to stall on; a second request would only
    # double its load.
    hedging = bool(config.get("transcription_hedging", False)) and not is_local_endpoint(endpoint)
    hedge_api_key = keys.next_key(config) if hedging else None
    second_pass = _second_pass_mode(config, model, operation_id)
    return TranscriptionRequest(
        api_key=api_key, api_endpoint=endpoint,
        audio_path=audio_path, prompt=sent_prompt(config),
        model=model, language=config["input_language"],
        temperature=config["transcription_temperature"],
        proxies=proxies_for(config),
        ffmpeg_path=None if recording_format is not None else resolve_ffmpeg(config.get("ffmpeg_path", "")),
        max_upload_bytes=int(config.get("max_upload_mb", 24) * 1024 * 1024),
        min_bitrate_kbps=int(config.get("min_audio_bitrate_kbps", 80)),
        recording_format=recording_format,
        recording_bitrate_kbps=int(config.get("recording_aac_bitrate_kbps", 64)),
        ffmpeg_setting=config.get("ffmpeg_path", ""),
        hedge_api_key=hedge_api_key,
        hedge_response_wait_s=float(config.get("transcription_hedging_wait_s", DEFAULT_HEDGE_RESPONSE_WAIT_S)),
        diagnostics=bool(config.get("transcription_diagnostics", True)),
        second_pass=second_pass,
        second_pass_threshold=float(config.get("transcription_second_pass_logprob", -0.5)),
        second_pass_prompt=config.get("transcription_second_pass_prompt", ""),
        second_pass_api_key=keys.next_key(config) or None if second_pass == "always" else None,
        local_engine_wait_s=target.local_wait_s,
        local_engine_error=target.local_error,
    )


def _second_pass_mode(config: Mapping[str, Any], model: str, operation_id: str) -> str:
    """The configured quality fallback, or "off" when the model reports no confidence to judge by."""
    second_pass = str(config.get("transcription_second_pass", "off"))
    if second_pass != "off" and not transcription_reports_confidence(model):
        logging.info("transcription_second_pass op=%s skipped reason=no_confidence_data model=%s",
                     operation_id, model)
        return "off"
    return second_pass
