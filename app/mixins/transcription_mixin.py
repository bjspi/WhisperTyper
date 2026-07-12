"""TranscriptionMixin — orchestrates transcription/rephrasing workers and result delivery.

Each in-flight request carries its own ``output_mode`` ("insert" types the result into the
focused field, "clipboard" copies it) through the signal chain, so concurrent requests
(e.g. a hotkey recording while a file transcription runs) can never clobber each other's
delivery mode.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import copykitten
from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QMessageBox

from app.core import liveprompt
from app.core.ffmpeg import is_video_file, resolve_ffmpeg
from app.core.textutil import shorten
from app.services.rephrasing_worker import RephrasingWorker
from app.services.transcription_worker import TranscriptionWorker


class TranscriptionMixin:
    """Transcription/rephrasing worker orchestration."""

    def start_transcription_worker(self, audio_path: str, output_mode: str = "insert") -> None:
        """Creates and starts a new thread for the transcription worker.

        Args:
            audio_path: Local path of the audio/video file to transcribe.
            output_mode: 'insert' types the result into the focused field; 'clipboard' copies
                it to the clipboard instead (better when no text field is focused yet).
        """
        # Resolve ffmpeg so video files get their audio extracted first; harmless for audio.
        ffmpeg_path = resolve_ffmpeg(self.config.get("ffmpeg_path", ""))
        needs_extraction = bool(ffmpeg_path) and is_video_file(audio_path)

        # Persistent spinner balloon: it stays until on_transcription_finished/error ends it,
        # so the hint tracks the real worker state instead of a fixed timeout. (timeout_ms is
        # ignored in spinner mode — a safety net inside the tooltip caps it.) For videos we open
        # on the "extracting…" phase; the worker emits `transcribing` to switch us to the second
        # spinner once ffmpeg is done and the upload begins.
        prefix = self._batch_progress_prefix()
        if needs_extraction:
            self.show_tray_balloon(
                prefix + self.translator.tr("extracting_video_audio_message", filename=os.path.basename(audio_path)),
                0, spinner=True,
            )
        else:
            self.show_tray_balloon(prefix + self.translator.tr("transcription_progress_message"), 0, spinner=True)

        thread = QThread()
        # The config now stores the language code directly
        lang_code = self.config["input_language"]

        worker = TranscriptionWorker(
            api_key=self.config["api_key"], api_endpoint=self.config["api_endpoint"],
            audio_path=audio_path, prompt=self.config["prompt"],
            model=self.config["model"], language=lang_code,
            temperature=self.config["transcription_temperature"],
            proxies=self._config_proxies(),
            ffmpeg_path=ffmpeg_path,
            max_upload_bytes=int(self.config.get("max_upload_mb", 24) * 1024 * 1024),
            min_bitrate_kbps=int(self.config.get("min_audio_bitrate_kbps", 80)),
        )
        worker.moveToThread(thread)
        self.active_workers.append(worker)
        self.active_threads.append(thread)

        thread.started.connect(worker.run)
        worker.compressing.connect(self._on_compression_phase_started)
        worker.transcribing.connect(self._on_transcription_phase_started)
        # Bind this request's output mode into the result handlers so a concurrently started
        # request (with a different mode) cannot redirect this one's delivery.
        worker.finished.connect(lambda text, mode=output_mode: self.on_transcription_finished(text, mode))
        worker.error.connect(self.on_transcription_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda t=thread, w=worker: self._cleanup_worker(t, w))
        thread.start()

    def _on_compression_phase_started(self, filename: str) -> None:
        """Show a distinct 'compressing…' spinner while an oversized file is downsampled.

        Emitted by the worker right before the (blocking) re-encode of a file that exceeds the
        upload limit, so this phase sits between the initial 'transcribing…' balloon and the
        actual upload. The worker then emits ``transcribing`` to switch back once the file fits.
        """
        self.show_tray_balloon(
            self._batch_progress_prefix() + self.translator.tr("compressing_audio_message", filename=filename),
            0, spinner=True,
        )

    def _on_transcription_phase_started(self) -> None:
        """Switch the spinner from the 'extracting…' phase to the 'transcribing…' phase.

        Emitted by the worker only after video extraction finishes, so the balloon reflects the
        two distinct phases instead of leaving 'extracting…' up during the whole upload.
        """
        self.show_tray_balloon(
            self._batch_progress_prefix() + self.translator.tr("transcription_progress_message"),
            0, spinner=True,
        )

    def _cleanup_worker(self, thread: QThread, worker: TranscriptionWorker) -> None:
        """Removes finished worker/thread references to allow garbage collection."""
        if worker in self.active_workers:
            self.active_workers.remove(worker)
        if thread in self.active_threads:
            self.active_threads.remove(thread)
        logging.info("Worker thread cleaned up.")

    def _cleanup_rephrasing_worker(self, thread: QThread, worker: RephrasingWorker) -> None:
        """Removes finished rephrasing worker/thread references."""
        if worker in self.active_rephrasing_workers:
            self.active_rephrasing_workers.remove(worker)
        if thread in self.active_rephrasing_threads:
            self.active_rephrasing_threads.remove(thread)
        logging.info("Rephrasing worker thread cleaned up.")

    def _start_rephrasing_worker(self, worker: RephrasingWorker, on_finished: object,
                                 on_error: object) -> None:
        """Move a prepared RephrasingWorker onto a fresh QThread and wire the standard plumbing."""
        thread = QThread()
        worker.moveToThread(thread)
        self.active_rephrasing_threads.append(thread)
        self.active_rephrasing_workers.append(worker)

        thread.started.connect(worker.run)
        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda t=thread, w=worker: self._cleanup_rephrasing_worker(t, w))
        thread.start()

    def _build_rephrasing_worker(self, system_prompt: str, user_prompt: str,
                                 context: str = "") -> RephrasingWorker:
        """Snapshot the current rephrasing API settings (GUI thread) into a self-contained worker."""
        return RephrasingWorker(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_url=self.config["rephrasing_api_url"],
            api_key=self.config["rephrasing_api_key"],
            model=self.config["rephrasing_model"],
            temperature=self.config["rephrasing_temperature"],
            context=context,
            proxies=self._config_proxies(),
        )

    # --- Batch file transcription (several picked files -> one joined clipboard result) ---
    def _start_batch_transcription(self, paths: list[str]) -> None:
        """Transcribe several files sequentially and join the results with blank lines.

        Each file runs through the normal transcription (+ optional rephrasing) pipeline; the
        per-file result is captured in ``_finalize_transcription_output`` instead of being copied,
        and the combined text lands on the clipboard once the last file is done. Empty or failing
        files are skipped so a single bad file never stalls the batch.
        """
        self._batch_files = list(paths)
        self._batch_index = 0
        self._batch_results: list[str] = []
        self._batch_active = True
        self._transcribe_next_in_batch()

    def _transcribe_next_in_batch(self) -> None:
        """Start the next queued file, or finish the batch when the queue is exhausted."""
        if self._batch_index >= len(self._batch_files):
            self._finish_batch_transcription()
            return
        # The spinner text for this file is prefixed with "[n/total] " via _batch_progress_prefix.
        self.start_transcription_worker(self._batch_files[self._batch_index], output_mode="clipboard")

    def _advance_batch(self, text: Optional[str]) -> None:
        """Record a finished file's text (if non-empty), then move on to the next file."""
        if text and text.strip():
            self._batch_results.append(text.strip())
        self._batch_index += 1
        self._transcribe_next_in_batch()

    def _finish_batch_transcription(self) -> None:
        """Join the collected results with blank lines, copy to clipboard, and reset batch state."""
        total = len(self._batch_files)
        done = len(self._batch_results)
        combined = "\n\n".join(self._batch_results)
        self._batch_active = False
        self._batch_files = []
        self._batch_results = []
        self._batch_index = 0
        if not combined:
            self.show_tray_balloon(self.translator.tr("no_speech_recognized_message"), 2500)
            return
        self.last_transcription = combined
        copykitten.copy(combined)
        self.show_tray_balloon(
            self.translator.tr("batch_transcribe_done_message", done=done, total=total),
            3500, check=True,
        )

    def _batch_progress_prefix(self) -> str:
        """A ``[n/total] `` prefix for spinner balloons while a batch runs (empty otherwise)."""
        if getattr(self, "_batch_active", False) and self._batch_files:
            return self.translator.tr(
                "batch_progress_prefix",
                current=self._batch_index + 1, total=len(self._batch_files),
            )
        return ""

    def on_transcription_finished(self, text: str, output_mode: str = "insert") -> None:
        """Handles a successful transcription result and routes it through rephrasing if enabled.

        Args:
            text: The transcribed text.
            output_mode: How this request's final text should be delivered (insert/clipboard).
        """
        processed = text.strip('"\'“”‘’ ')
        prompt = self.config["prompt"].strip()
        if not processed or (prompt and processed.lower() == prompt.lower()):
            if getattr(self, "_batch_active", False):
                logging.info("Batch file produced no usable speech; skipping it.")
                self._advance_batch(None)
                return
            self.show_tray_balloon(self.translator.tr("no_speech_recognized_message"), 2000)
            logging.info("Transcription result was empty or matched the prompt, ignoring.")
            return

        # --- Rephrasing (runs in a worker thread so the spinner keeps animating) ---
        # 1. LivePrompting via trigger words: the transcription itself is the instruction.
        if self.config["liveprompt_enabled"]:
            trigger_words = liveprompt.parse_trigger_words(self.config.get("liveprompt_trigger_words", ""))
            scan_depth = self.config.get("liveprompt_trigger_word_scan_depth", 5)

            if liveprompt.contains_trigger(processed, trigger_words, scan_depth):
                # Optionally drop the trigger word and everything before it, so only the
                # actual instruction after it is sent to the model.
                instruction = processed
                if self.config.get("liveprompt_strip_trigger", False):
                    instruction = liveprompt.strip_trigger(processed, trigger_words)
                self._start_post_transcription_rephrase(
                    system_prompt=self.config["liveprompt_system_prompt"],
                    user_prompt=instruction,
                    context=self.current_transcription_context if self.config["rephrase_use_selection_context"] else "",
                    original_text=processed,
                    output_mode=output_mode,
                )
                return

        # 2. Generic rephrasing: combine the generic prompt with the transcription.
        if self.config["generic_rephrase_enabled"]:
            self._start_post_transcription_rephrase(
                system_prompt="",  # System prompt is not used here in the same way
                user_prompt=f"{self.config['generic_rephrase_prompt']}\n\nText: {processed}",
                context="",  # Context is not used for generic rephrasing
                original_text=processed,
                output_mode=output_mode,
            )
            return

        # 3. No rephrasing: deliver the raw transcription.
        self._finalize_transcription_output(processed, output_mode=output_mode)

    def _start_post_transcription_rephrase(self, system_prompt: str, user_prompt: str,
                                           context: str, original_text: str,
                                           output_mode: str = "insert") -> None:
        """Run the transcription's rephrasing/LivePrompt request in a worker thread.

        The persistent spinner balloon stays up for the whole request (analogous to the
        transcription spinner) and is ended by the finished/error callbacks. Running off the GUI
        thread keeps the UI responsive and lets the spinner actually animate.
        """
        # Persistent spinner while the rephrasing request runs (timeout_ms ignored in spinner mode).
        # Preview the FINAL prompt actually sent (after trigger stripping), shortened to keep the
        # balloon compact.
        self.show_tray_balloon(
            self._batch_progress_prefix() + self.translator.tr(
                "rephrasing_transcript_message",
                processed_text=shorten(user_prompt),
            ),
            0, spinner=True,
        )

        worker = self._build_rephrasing_worker(system_prompt, user_prompt, context)
        self._start_rephrasing_worker(
            worker,
            on_finished=lambda rt, mode=output_mode: self._finalize_transcription_output(
                rt, spinner_active=True, output_mode=mode),
            on_error=lambda em, orig=original_text, mode=output_mode:
                self._on_post_transcription_rephrase_error(em, orig, mode),
        )

    def _on_post_transcription_rephrase_error(self, error_message: str, original_text: str,
                                              output_mode: str = "insert") -> None:
        """Rephrasing failed: fall back to the raw transcription and end the spinner."""
        logging.error(f"Post-transcription rephrasing failed: {error_message}")
        if "empty text" in error_message:
            # Empty result is not a hard failure — silently deliver the raw transcription and let
            # _finalize end the still-active spinner.
            self._finalize_transcription_output(original_text, spinner_active=True, output_mode=output_mode)
        else:
            # A real failure: replace the spinner with an error notice, then deliver raw text
            # without hiding that notice.
            self.show_tray_balloon(self.translator.tr("rephrasing_failed_message", error=error_message), 3000)
            self._finalize_transcription_output(original_text, spinner_active=False, output_mode=output_mode)

    def _finalize_transcription_output(self, text: str, spinner_active: bool = True,
                                       output_mode: str = "insert") -> None:
        """Deliver the final text (insert or clipboard) and end the spinner balloon.

        Args:
            text: The text to insert/copy.
            spinner_active: Whether the persistent spinner is still showing and must be hidden.
                False when a caller already replaced it with a normal balloon (e.g. an error).
            output_mode: 'insert' types the text into the focused field; 'clipboard' copies it.
        """
        self.last_transcription = text
        # In a batch, capture this file's text and trigger the next one instead of delivering now;
        # the joined result is copied once the whole batch finishes.
        if getattr(self, "_batch_active", False):
            self._advance_batch(text)
            return
        if output_mode == "clipboard":
            # Copy instead of type — used for tray re-transcribe / file transcription, where the
            # user hasn't focused a text field. Confirm with a green checkmark balloon (this also
            # replaces the persistent "transcribing…" spinner).
            copykitten.copy(text)
            self.show_tray_balloon(self.translator.tr("transcribed_to_clipboard_message"), 2500, check=True)
        else:
            # Insert mode: swap the spinner for a brief "done ✓" balloon, then type the text. If
            # the spinner was already replaced by an error notice (spinner_active=False), leave
            # that notice alone rather than clobbering it with a success checkmark.
            if spinner_active:
                self.show_tray_balloon(self.translator.tr("transcription_done_message"), 1600, check=True)
            self.insert_transcribed_text(text)

    def on_transcription_error(self, error_message: str, audio_file_path: str) -> None:
        """Handles errors that occur during transcription."""
        logging.error(f"Transcription error: {error_message}")
        # In a batch, don't block on a modal retry dialog — log, skip this file, and keep going.
        if getattr(self, "_batch_active", False):
            logging.warning("Batch file failed, skipping: %s", audio_file_path)
            self._advance_batch(None)
            return
        self.show_tray_balloon(self.translator.tr("transcription_failed_message"), 4000)
        self._set_idle_tray_icon()

        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Warning)
        msg_box.setText(self.translator.tr("transcription_error_title"))
        msg_box.setInformativeText(self.translator.tr("transcription_error_text", error_message=error_message, audio_file_path=audio_file_path))
        msg_box.setWindowTitle(self.translator.tr("transcription_error_title"))
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Retry)
        if msg_box.exec() == QMessageBox.StandardButton.Retry:
            self.start_transcription_worker(audio_file_path)

    def _drain_worker_threads(self) -> None:
        """Ask any in-flight transcription/rephrasing QThreads to finish before exit.

        A bare QThread that is still running an HTTP request when the interpreter tears
        down triggers Qt's "QThread: Destroyed while thread is still running" abort. We
        request quit() and give each a short grace period to unwind.
        """
        for registry in (getattr(self, 'active_threads', []), getattr(self, 'active_rephrasing_threads', [])):
            for thread in list(registry):
                try:
                    if thread.isRunning():
                        thread.quit()
                        thread.wait(2000)
                except Exception as e:
                    logging.debug(f"Worker thread drain skipped one thread: {e}")
