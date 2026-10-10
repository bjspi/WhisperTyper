"""Transcription and rephrasing requests: worker orchestration, routing and result delivery.

Each in-flight request carries its own ``output_mode`` ("insert" types the result into the
focused field, "clipboard" copies it) through the signal chain, so concurrent requests
(e.g. a hotkey recording while a file transcription runs) can never clobber each other's
delivery mode.
"""
from __future__ import annotations

import logging
import os
from typing import Callable, List, Optional

import copykitten
from PyQt6.QtWidgets import QMessageBox

from app.context import AppContext
from app.controllers.text_output import TextOutput
from app.controllers.warmup import WarmupScheduler
from app.controllers.workers import WorkerThreads
from app.core.api_keys import GroqKeyRotation, provider_for_url, selected_api_key
from app.core.audio_formats import is_video_file, needs_audio_normalization
from app.core.constants import LANGUAGES
from app.core.rephrase_routing import RephrasePlan, plan_rephrase, usable_transcript
from app.core.replacements import ReplacementError, Replacements
from app.core.textutil import shorten
from app.core.timing import NO_TIMING, OperationTiming
from app.services.ffmpeg import resolve_ffmpeg
from app.services.net import proxies_for
from app.services.rephrasing_worker import RephrasingWorker
from app.services.transcription import TranscriptionRequest
from app.services.transcription_worker import TranscriptionWorker
from app.ui.durations import (
    BALLOON_CONFIRM_MS,
    BALLOON_ERROR_MS,
    BALLOON_INFO_MS,
    BALLOON_NOTICE_MS,
    BALLOON_SHORT_MS,
    BALLOON_WARNING_MS,
)


class TranscriptionPipeline:
    """Transcribe audio, route the transcript through rephrasing and deliver the final text."""

    def __init__(self, ctx: AppContext, output: TextOutput, warmup: WarmupScheduler, workers: WorkerThreads) -> None:
        """Requests share the warm HTTP pool (``warmup``) and the thread registry (``workers``)."""
        self._ctx = ctx
        self._config = ctx.config
        self._notifier = ctx.notifier
        self._output = output
        self._warmup = warmup
        self._workers = workers
        self._key_rotation = GroqKeyRotation()
        #: Final text of the latest delivered transcription (tray copy action, double-click).
        self.last_transcription = ""
        #: Selected text captured when the current recording started (LivePrompt context).
        self.selection_context = ""
        self.replacement_rules = Replacements("")
        self.reload_replacements()
        self._batch_files: List[str] = []
        self._batch_index = 0
        self._batch_results: List[str] = []
        self._batch_active = False

    def reload_replacements(self) -> None:
        """Compile the saved correction rules; invalid rules disable corrections until fixed."""
        try:
            self.replacement_rules = Replacements(self._config["replacements_rules"])
        except ReplacementError as error:
            self.replacement_rules = Replacements("")
            logging.warning("replacements_config_invalid line=%s; corrections disabled until settings are fixed", error.line)

    # --- Transcription -------------------------------------------------------------------
    def start(self, audio_path: str, output_mode: str = "insert",
              transformation_prompt: Optional[str] = None, timing: Optional[OperationTiming] = None) -> None:
        """Transcribe ``audio_path`` on a worker thread.

        Args:
            audio_path: Local path of the audio/video file to transcribe.
            output_mode: 'insert' types the result into the focused field; 'clipboard' copies
                it to the clipboard instead (better when no text field is focused yet).
            transformation_prompt: Optional system prompt selected for this microphone recording.
            timing: Recording-stop timing, or a fresh operation for a file transcription.
        """
        config = self._config
        timing = timing or OperationTiming()
        timing.mark("transcription_queued")
        self._warmup.touch()
        # Retained microphone WAVs (including retries) use the selected recording format; the worker
        # resolves ffmpeg only if a long recording still exceeds the upload limit.
        recording_format = config.get("recording_format", "wav") if self._ctx.recordings.owns(audio_path) else None
        ffmpeg_path = None if recording_format is not None else resolve_ffmpeg(config.get("ffmpeg_path", ""))
        extracts_video = bool(ffmpeg_path) and is_video_file(audio_path)
        converts_audio = bool(ffmpeg_path) and needs_audio_normalization(audio_path)
        # Snapshot the language together with the worker settings. A later settings change must
        # not make the status balloon disagree with the language used by this in-flight request.
        lang_code = config["input_language"]

        # Persistent spinner balloon: it stays until the result/error handler ends it, so the hint
        # tracks the real worker state instead of a fixed timeout. For videos it opens on the
        # "extracting…"/"converting…" phase; the worker emits `transcribing` once ffmpeg is done.
        prefix = self._batch_progress_prefix()
        if extracts_video or converts_audio:
            key = "extracting_video_audio_message" if extracts_video else "converting_audio_message"
            self._notifier.show(prefix + self._ctx.tr(key, filename=os.path.basename(audio_path)), 0, spinner=True)
        else:
            self._notifier.show(prefix + self._transcription_progress_message(lang_code), 0, spinner=True)

        api_key = self._key_rotation.next_key(config)
        provider = provider_for_url(config["api_endpoint"])
        logging.info("transcription_credential op=%s provider=%s profile_id=%s rotation=%s",
                     timing.operation_id, provider, self._key_rotation.last_profile_id,
                     bool(config["groq_key_rotation"] and provider == "groq"))
        request = TranscriptionRequest(
            api_key=api_key, api_endpoint=config["api_endpoint"],
            audio_path=audio_path, prompt=config["prompt"],
            model=config["model"], language=lang_code,
            temperature=config["transcription_temperature"],
            proxies=proxies_for(config),
            ffmpeg_path=ffmpeg_path,
            max_upload_bytes=int(config.get("max_upload_mb", 24) * 1024 * 1024),
            min_bitrate_kbps=int(config.get("min_audio_bitrate_kbps", 80)),
            recording_format=recording_format,
            recording_bitrate_kbps=int(config.get("recording_aac_bitrate_kbps", 64)),
            ffmpeg_setting=config.get("ffmpeg_path", ""),
        )
        worker = TranscriptionWorker(request, timing=timing, tr=self._ctx.tr)
        worker.compressing.connect(self._on_compression_phase_started)
        worker.transcribing.connect(lambda language=lang_code: self._on_transcription_phase_started(language))
        # Bind this request's output mode into the result handlers so a concurrently started
        # request (with a different mode) cannot redirect this one's delivery.
        worker.finished.connect(
            lambda text, mode=output_mode, selected_prompt=transformation_prompt, operation=timing:
                self.on_transcription_finished(text, mode, selected_prompt, operation)
        )
        worker.error.connect(
            lambda message, path, mode=output_mode, selected_prompt=transformation_prompt, operation=timing:
                self.on_transcription_error(message, path, mode, selected_prompt, operation)
        )
        self._workers.start(worker, (worker.finished, worker.error),
                            queued=lambda: timing.mark("transcription_worker_queued"))

    def _on_compression_phase_started(self, filename: str) -> None:
        """Show a distinct 'compressing…' spinner while an oversized file is downsampled."""
        self._notifier.show(self._batch_progress_prefix() + self._ctx.tr("compressing_audio_message", filename=filename),
                            0, spinner=True)

    def _transcription_progress_message(self, lang_code: str) -> str:
        """Return the localized spinner text with the configured language display name."""
        language_name = next(
            (name for name, code in LANGUAGES.items() if code == lang_code),
            lang_code or "Detect Language",
        )
        return self._ctx.tr("transcription_progress_message", language=language_name)

    def _on_transcription_phase_started(self, lang_code: str) -> None:
        """Switch the spinner from the 'extracting…' phase to the 'transcribing…' phase."""
        self._notifier.show(self._batch_progress_prefix() + self._transcription_progress_message(lang_code),
                            0, spinner=True)

    def on_transcription_finished(self, text: str, output_mode: str = "insert",
                                  transformation_prompt: Optional[str] = None,
                                  timing: Optional[OperationTiming] = None) -> None:
        """Handle a successful transcription and route it through rephrasing if configured.

        Args:
            text: The transcribed text.
            output_mode: How this request's final text should be delivered (insert/clipboard).
            transformation_prompt: Explicit recording-palette prompt, if one was selected.
            timing: This request's recording/HTTP timings, preserved through rephrasing.
        """
        timing = timing or OperationTiming("transcription_result")
        timing.mark("transcription_result_received")
        processed = usable_transcript(text, self._config["prompt"])
        if processed is None:
            timing.finish("no_speech")
            if self._batch_active:
                logging.info("Batch file produced no usable speech; skipping it.")
                self._advance_batch(None)
                return
            self._notifier.show(self._ctx.tr("no_speech_recognized_message"), BALLOON_SHORT_MS)
            logging.info("Transcription result was empty or matched the prompt, ignoring.")
            return

        processed = self._apply_replacements(processed, timing)
        timing.mark("result_processed")
        plan = plan_rephrase(processed, self._config, transformation_prompt, self.selection_context)
        if plan is None:
            self.finalize_output(processed, output_mode=output_mode, timing=timing)
        else:
            self._start_post_transcription_rephrase(plan, processed, output_mode, timing)

    def _apply_replacements(self, text: str, timing: OperationTiming) -> str:
        """Apply the saved correction rules; logs counts and rule lines, never transcript text."""
        rules = self.replacement_rules
        enabled = self._config["replacements_enabled"]
        matches = 0
        matched_lines: list[int] = []
        with timing.span("replacements"):
            if enabled:
                text, matches, matched_lines = rules.apply(text)
        logging.info("replacements_check op=%s enabled=%s rules=%s terms=%s text_chars=%s matches=%s matched_rules=%s",
                     timing.operation_id, enabled, rules.rule_count, rules.term_count, len(text), matches, matched_lines)
        return text

    def on_transcription_error(self, error_message: str, audio_file_path: str,
                               output_mode: str = "insert",
                               transformation_prompt: Optional[str] = None,
                               timing: OperationTiming = NO_TIMING) -> None:
        """Report a failed transcription and offer a retry (batches skip the file instead)."""
        logging.error(f"Transcription error: {error_message}")
        timing.finish("transcription_failed")
        # In a batch, don't block on a modal retry dialog — log, skip this file, and keep going.
        if self._batch_active:
            logging.warning("Batch file failed, skipping: %s", audio_file_path)
            self._advance_batch(None)
            return
        self._notifier.show(self._ctx.tr("transcription_failed_message"), BALLOON_WARNING_MS)

        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Warning)
        msg_box.setText(self._ctx.tr("transcription_error_title"))
        msg_box.setInformativeText(self._ctx.tr("transcription_error_text", error_message=error_message,
                                                audio_file_path=audio_file_path))
        msg_box.setWindowTitle(self._ctx.tr("transcription_error_title"))
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Retry)
        if msg_box.exec() == QMessageBox.StandardButton.Retry:
            self.start(audio_file_path, output_mode=output_mode, transformation_prompt=transformation_prompt,
                       timing=OperationTiming("retry"))

    # --- Rephrasing ----------------------------------------------------------------------
    def start_rephrasing(self, system_prompt: str, user_prompt: str, context: str, timing: Optional[OperationTiming],
                         on_finished: Callable[[str, OperationTiming], None],
                         on_empty: Callable[[OperationTiming], None],
                         on_error: Callable[[str, OperationTiming], None]) -> None:
        """Snapshot the rephrasing API settings into a worker and run it; handlers get the timing."""
        timing = timing or OperationTiming("rephrase")
        timing.mark("rephrase_queued")
        self._warmup.touch()
        worker = RephrasingWorker(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_url=self._config["rephrasing_api_url"],
            api_key=selected_api_key(self._config, "rephrasing"),
            model=self._config["rephrasing_model"],
            temperature=self._config["rephrasing_temperature"],
            context=context,
            proxies=proxies_for(self._config),
            timing=timing,
        )
        operation = worker.timing
        worker.finished.connect(lambda text: on_finished(text, operation))
        worker.empty.connect(lambda: on_empty(operation))
        worker.error.connect(lambda message: on_error(message, operation))
        self._workers.start(worker, (worker.finished, worker.empty, worker.error),
                            queued=lambda: operation.mark("rephrase_worker_queued"))

    def _start_post_transcription_rephrase(self, plan: RephrasePlan, original_text: str,
                                           output_mode: str, timing: OperationTiming) -> None:
        """Run the planned rephrasing/LivePrompt request; the spinner stays up until it ends."""
        # Preview the FINAL prompt actually sent (after trigger stripping), shortened to keep the
        # balloon compact. The spinner ignores its timeout.
        self._notifier.show(self._batch_progress_prefix() + self._ctx.tr(
            "rephrasing_transcript_message", processed_text=shorten(plan.user_prompt)), 0, spinner=True)

        def deliver_original(spinner_active: bool, operation: OperationTiming) -> None:
            """Fall back to the unmodified transcription."""
            self.finalize_output(original_text, spinner_active=spinner_active, output_mode=output_mode,
                                 timing=operation, outcome="rephrase_failed_fallback")

        def on_error(error_message: str, operation: OperationTiming) -> None:
            """Replace the spinner with the error notice, then deliver the raw text without hiding it."""
            logging.error("Post-transcription rephrasing failed: %s", error_message)
            self._notifier.show(self._ctx.tr("rephrasing_failed_message", error=error_message), BALLOON_ERROR_MS)
            deliver_original(False, operation)

        self.start_rephrasing(
            plan.system_prompt, plan.user_prompt, plan.context, timing,
            on_finished=lambda text, operation: self.finalize_output(
                text, spinner_active=True, output_mode=output_mode, timing=operation),
            # An empty answer is no hard failure: deliver the raw text and let finalize end the spinner.
            on_empty=lambda operation: deliver_original(True, operation),
            on_error=on_error,
        )

    # --- Delivery ------------------------------------------------------------------------
    def finalize_output(self, text: str, spinner_active: bool = True, output_mode: str = "insert",
                        timing: OperationTiming = NO_TIMING, outcome: str = "ok") -> None:
        """Deliver the final text (insert or clipboard) and end the spinner balloon.

        Args:
            text: The text to insert/copy.
            spinner_active: Whether the persistent spinner is still showing and must be hidden.
                False when a caller already replaced it with a normal balloon (e.g. an error).
            output_mode: 'insert' types the text into the focused field; 'clipboard' copies it.
            timing: The operation whose text is being delivered.
            outcome: Distinguishes normal completion from a rephrasing fallback.
        """
        self.last_transcription = text
        # In a batch, capture this file's text and trigger the next one instead of delivering now;
        # the joined result is copied once the whole batch finishes.
        if self._batch_active:
            timing.finish("batch_buffered")
            self._advance_batch(text)
            return
        timing.measure_output(lambda: self._deliver(text, spinner_active, output_mode, timing), outcome)

    def _deliver(self, text: str, spinner_active: bool, output_mode: str, timing: OperationTiming) -> bool:
        """Copy or insert the final text; returns whether dispatch succeeded."""
        if output_mode == "clipboard":
            # Copy instead of type — used for tray re-transcribe / file transcription, where the
            # user hasn't focused a text field. The checkmark balloon replaces the spinner.
            with timing.span("clipboard_write"):
                copykitten.copy(text)
            self._notifier.show(self._ctx.tr("transcribed_to_clipboard_message"), BALLOON_INFO_MS, check=True)
            return True
        # Insert mode: swap the spinner for a brief "done ✓" balloon, then type the text. If
        # the spinner was already replaced by an error notice (spinner_active=False), leave
        # that notice alone rather than clobbering it with a success checkmark.
        if spinner_active:
            self._notifier.show(self._ctx.tr("transcription_done_message"), BALLOON_CONFIRM_MS, check=True)
        return self._output.insert(text, timing=timing)

    def copy_last_transcription(self) -> None:
        """Copy the last delivered text to the clipboard again."""
        if not self.last_transcription:
            self._notifier.show(self._ctx.tr("no_transcription_to_copy_message"), BALLOON_SHORT_MS)
            return
        copykitten.copy(self.last_transcription)
        self._notifier.show(self._ctx.tr("transcription_copied_message"), BALLOON_SHORT_MS)

    # --- Batch file transcription (several picked files -> one joined clipboard result) ---
    def start_batch(self, paths: List[str]) -> None:
        """Transcribe several files sequentially and join the results with blank lines.

        Each file runs through the normal transcription (+ optional rephrasing) pipeline; the
        per-file result is captured in ``finalize_output`` instead of being copied, and the
        combined text lands on the clipboard once the last file is done. Empty or failing files
        are skipped so a single bad file never stalls the batch.
        """
        self._batch_files = list(paths)
        self._batch_index = 0
        self._batch_results = []
        self._batch_active = True
        self._transcribe_next_in_batch()

    def _transcribe_next_in_batch(self) -> None:
        """Start the next queued file, or finish the batch when the queue is exhausted."""
        if self._batch_index >= len(self._batch_files):
            self._finish_batch()
            return
        self.start(self._batch_files[self._batch_index], output_mode="clipboard")

    def _advance_batch(self, text: Optional[str]) -> None:
        """Record a finished file's text (if non-empty), then move on to the next file."""
        if text and text.strip():
            self._batch_results.append(text.strip())
        self._batch_index += 1
        self._transcribe_next_in_batch()

    def _finish_batch(self) -> None:
        """Join the collected results with blank lines, copy to clipboard, and reset batch state."""
        total = len(self._batch_files)
        done = len(self._batch_results)
        combined = "\n\n".join(self._batch_results)
        self._batch_active = False
        self._batch_files = []
        self._batch_results = []
        self._batch_index = 0
        if not combined:
            self._notifier.show(self._ctx.tr("no_speech_recognized_message"), BALLOON_INFO_MS)
            return
        self.last_transcription = combined
        copykitten.copy(combined)
        self._notifier.show(self._ctx.tr("batch_transcribe_done_message", done=done, total=total),
                            BALLOON_NOTICE_MS, check=True)

    def _batch_progress_prefix(self) -> str:
        """A ``[n/total] `` prefix for spinner balloons while a batch runs (empty otherwise)."""
        if self._batch_active and self._batch_files:
            return self._ctx.tr("batch_progress_prefix", current=self._batch_index + 1, total=len(self._batch_files))
        return ""
