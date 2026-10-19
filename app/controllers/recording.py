"""Recording lifecycle: start/stop/cancel, capture strategy, prompt palette and hand-off to transcription."""
from __future__ import annotations

import logging
import time
import wave
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Union

from PyQt6.QtCore import QObject, QPoint, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication

from app.audio import macos_recorder
from app.audio.capture import HotMicCapture, MicrophoneInput, OnDemandCapture
from app.audio.sound import SoundPlayer
from app.context import AppContext
from app.controllers.permissions import MacPermissions
from app.controllers.text_output import TextOutput
from app.controllers.transcription import TranscriptionPipeline
from app.controllers.warmup import WarmupScheduler
from app.core import dsp
from app.core.api_keys import rephrasing_configured, transcription_configured
from app.core.env import is_MACOS, is_WINDOWS
from app.core.prompts import INSTRUCTION, auto_apply_prompt, instruction_entry, recording_prompt_entries
from app.core.redaction import redact_for_log
from app.core.timing import OperationTiming
from app.platform.windows import is_console_foreground_window
from app.ui import theme
from app.ui.durations import (
    BALLOON_ERROR_MS,
    BALLOON_INFO_MS,
    BALLOON_PERSISTENT_MS,
    BALLOON_SHORT_MS,
    BALLOON_WARNING_MS,
)
from app.ui.floating_buttons import RecordingPromptOverlay
from app.ui.tooltip import MouseFollowerTooltip

#: Sample rate of uploaded recordings.
UPLOAD_SAMPLERATE = 16000
#: How long the microphone may take to deliver its first block before the plain balloon shows.
MIC_SIGNAL_GRACE_MS = 1500

#: PyAudio frames per read.
CHUNK_SIZE = 1024
#: Waveform bars per captured block (1024 frames at 16 kHz: one bar per 32 ms).
WAVE_BARS_PER_BLOCK = 2


class RecordingController(QObject):
    """Record from the microphone and hand the retained WAV to the transcription pipeline."""

    #: True when a recording starts, False when it stops, is cancelled or fails to start.
    state_changed = pyqtSignal(bool)
    # The capture thread reports that the stream could not be opened for a take; the recording ends on the GUI thread.
    _capture_failed = pyqtSignal(int)
    # The capture thread reports the first recorded block of a take; the palette opens on the GUI thread.
    _recording_audio = pyqtSignal(int)

    def __init__(self, ctx: AppContext, *, permissions: MacPermissions, warmup: WarmupScheduler,
                 output: TextOutput, pipeline: TranscriptionPipeline, sounds: SoundPlayer,
                 open_settings: Callable[[], None], palette_anchor: Callable[[], Optional[QPoint]]) -> None:
        """``palette_anchor`` returns the tray icon position the prompt palette is placed next to."""
        super().__init__()
        self._ctx = ctx
        self._config = ctx.config
        self._notifier = ctx.notifier
        self._permissions = permissions
        self._warmup = warmup
        self._output = output
        self._pipeline = pipeline
        self._sounds = sounds
        self._open_settings = open_settings
        self._palette_anchor = palette_anchor
        self.is_recording = False
        #: True while the running recording is paused from the palette; stopping still transcribes.
        self.is_paused = False
        #: Set while the push-to-talk key is held; the hotkey listener reads and clears it.
        self.push_to_talk_active = False
        #: Prompt explicitly clicked in the palette for the running recording.
        self.current_prompt: Optional[str] = None
        #: True when that click was the instruction entry: the dictation is carried out as an order.
        self.current_prompt_is_instruction = False
        #: False once "None" is clicked: the automatic prompt is skipped for this recording.
        self.use_auto_prompt = True
        self._palette_status: Optional[MouseFollowerTooltip] = None
        #: Counts started recordings, so a late "audio arrived" report of an older one is ignored.
        self._take = 0
        #: True until the running recording has shown its balloon or prompt palette.
        self._feedback_pending = False
        self.latest_audio_level = 0.0
        #: Recent block levels in dBFS for the palette's waveform; appended by the capture thread.
        self._wave_levels: deque[float] = deque(maxlen=256)
        self._mac_device_name = ""
        self.last_activity_ts = time.monotonic()
        self.microphone = MicrophoneInput(lambda: self._config, UPLOAD_SAMPLERATE, CHUNK_SIZE,
                                          self._notify_input_device_fallback)
        self.hot_mic = HotMicCapture(self.microphone, idle_expired=self._hot_mic_idle_expired,
                                     on_block=self._update_latest_audio_level,
                                     on_recording_audio=self._report_recording_audio,
                                     on_failure=self._on_capture_open_failed)
        self.on_demand = OnDemandCapture(self.microphone, on_block=self._update_latest_audio_level,
                                         on_failure=self._on_capture_open_failed,
                                         on_recording_audio=self._report_recording_audio)
        self.mac_recorder = macos_recorder.MacRecorder()
        self._capture_failed.connect(self._on_capture_failed)
        self._recording_audio.connect(self._on_recording_audio)

    # --- Capture strategy ----------------------------------------------------------------
    def _update_latest_audio_level(self, pcm_chunk: bytes) -> None:
        """Feed the tray meter and the palette's waveform from a captured block (capture thread)."""
        self.latest_audio_level = dsp.meter_level(pcm_chunk)
        self._wave_levels.extend(dsp.block_dbfs(pcm_chunk, WAVE_BARS_PER_BLOCK))

    def drain_wave_levels(self) -> List[float]:
        """Levels in dBFS captured since the last call, for the palette's waveform (GUI thread)."""
        if self.mac_recorder.active:
            power = self.mac_recorder.power_dbfs()
            return [] if power is None else [power]
        levels = []
        while self._wave_levels:
            levels.append(self._wave_levels.popleft())
        return levels

    def input_device_display_name(self) -> str:
        """The microphone the running recording comes from ("" while the stream is not open yet)."""
        if self.mac_recorder.active:
            if not self._mac_device_name:
                self._mac_device_name = macos_recorder.default_input_name()
            return self._mac_device_name
        return self.microphone.device_name

    def _device_menu_entries(self) -> List[tuple]:
        """(label, config value) pairs for the palette's microphone menu."""
        entries = [(self._ctx.tr("input_device_default"), "")]
        entries += [(device["name"], device["name"]) for device in self.selectable_input_devices()]
        return entries

    def switch_input_device(self, name: str) -> None:
        """Record from another microphone, picked in the palette: the take starts over on it.

        Audio from two devices (possibly at different sample rates) cannot be joined, so the
        take so far is discarded, like the restart button does.
        """
        logging.info(f"Input device switch requested: {name or 'System Default'!r} "
                     f"(recording={self.is_recording}, was {self.microphone.device_name!r}).")
        self._config["input_device_name"] = name
        self._ctx.save_config()
        if not self.is_recording or self.mac_recorder.active:
            self.apply_input_device_selection()
            return
        self.microphone.device_name = ""
        if self.keep_mic_hot():
            # Join the reader before the stream closes; the recording flag survives the restart.
            self.hot_mic.stop()
            self.hot_mic.restart_recording()
            self.hot_mic.start()
        else:
            self.on_demand.cancel_recording()
            self.on_demand.start_recording()
        self._wave_levels.clear()
        self.is_paused = False
        overlay = RecordingPromptOverlay._instance
        if overlay is not None:
            overlay.set_paused(False)
            overlay.restart_timer()
        logging.info("Take restarted on the new input device; the stream opens on the reader thread.")

    def level(self) -> float:
        """Current input level for the tray meter (the macOS recorder meters natively)."""
        native_level = self.mac_recorder.level()
        if native_level is not None:
            self.latest_audio_level = native_level
        return self.latest_audio_level

    def _on_capture_open_failed(self) -> None:
        """The stream could not be opened for the take (capture thread): hand over to the GUI thread."""
        self._capture_failed.emit(self._take)

    def _on_capture_failed(self, take: int) -> None:
        """The microphone cannot serve this take: end the recording like a failed start does."""
        if take != self._take or not self.is_recording or self.mac_recorder.active:
            return
        logging.error("Recording ended: the audio input stream could not be opened.")
        self.is_recording = False
        self.is_paused = False
        self.push_to_talk_active = False
        self._feedback_pending = False
        self.abandon_prompt_selection()
        self.active_capture().cancel_recording()
        self.state_changed.emit(False)
        self._notifier.show(self._ctx.tr("no_microphone_signal_message"), BALLOON_INFO_MS)

    def _report_recording_audio(self) -> None:
        """The first block of the take was recorded (capture thread): hand over to the GUI thread."""
        self._recording_audio.emit(self._take)

    def _notify_input_device_fallback(self, device: str) -> None:
        """Tell the user the configured device failed and the default is used (capture thread; thread-safe)."""
        self._notifier.show(self._ctx.tr("input_device_fallback_message", device=device), BALLOON_WARNING_MS)

    def keep_mic_hot(self) -> bool:
        """Return whether the Windows background microphone prewarm mode is enabled."""
        return is_WINDOWS and self._config.get("windows_keep_mic_hot", True)

    def active_capture(self) -> Union[HotMicCapture, OnDemandCapture]:
        """The PyAudio capture strategy for the current settings."""
        return self.hot_mic if self.keep_mic_hot() else self.on_demand

    def start_background_capture(self) -> None:
        """Start the Windows background mic reader to avoid clipped leading audio."""
        if not is_WINDOWS or self.hot_mic.running:
            return
        self.touch_activity()
        self.hot_mic.start()

    def stop_background_capture(self) -> None:
        """Stop the Windows background mic reader and release the capture device."""
        # A reader stuck in a driver read still owns the stream; terminating PortAudio under it would crash.
        if self.hot_mic.stop():
            self.microphone.terminate()

    def apply_settings(self) -> None:
        """Start or stop the warm microphone after the settings were saved."""
        if self.keep_mic_hot():
            self.touch_activity()
            self.start_background_capture()
        else:
            self.stop_background_capture()

    def shutdown(self) -> None:
        """Release every capture resource at quit."""
        self.stop_background_capture()
        if self.mac_recorder.active:
            self.mac_recorder.stop(discard=True)

    def probe_microphone(self) -> None:
        """Open and close the microphone once (triggers the macOS permission prompt)."""
        if macos_recorder.available():
            self.mac_recorder.start(self._ctx.recordings.new_path())
            self.mac_recorder.stop(discard=True)
        else:
            self.microphone.open()
            self.microphone.close()

    def touch_activity(self) -> None:
        """Update the timestamp used for the warm microphone's idle timeout."""
        self.last_activity_ts = time.monotonic()

    def _hot_mic_idle_expired(self) -> bool:
        """Whether the configured prewarm idle timeout elapsed since the last activity (0 = never)."""
        idle_seconds = max(0, int(self._config.get("windows_keep_mic_hot_idle_minutes", 15))) * 60
        return idle_seconds > 0 and time.monotonic() - self.last_activity_ts >= idle_seconds

    def selectable_input_devices(self) -> List[Dict[str, Any]]:
        """Input devices on the platform's default host API, for the settings dropdown."""
        return self.microphone.devices.selectable_input_devices()

    def apply_input_device_selection(self) -> None:
        """Reopen the capture stream on the newly selected input device (thread-safe).

        Closing a PortAudio stream while the background reader thread is inside stream.read()
        segfaults, so the reader is stopped (joined) BEFORE the stream is closed, then restarted.
        """
        if self.is_recording:
            # Don't yank the stream out from under an active recording.
            logging.info("Input device change deferred until the current recording stops.")
            return
        reader_running = self.hot_mic.running
        if reader_running:
            self.stop_background_capture()  # joins the reader, then closes the stream
        else:
            self.microphone.close()
        if reader_running and self.keep_mic_hot():
            self.start_background_capture()
        logging.info(
            "Input device set to "
            f"{self._config.get('input_device_name') or 'System Default'!r}; capture stream reset."
        )

    # --- Lifecycle -----------------------------------------------------------------------
    def toggle(self, *, detected_ns: Optional[int] = None) -> None:
        """Start or stop recording; ``detected_ns`` is the hotkey listener's detection time."""
        if self.is_recording:
            self._stop(OperationTiming("recording", detected_ns))
        else:
            self._start()

    def _start(self) -> None:
        """Start capturing, after checking that a transcription could be sent at all."""
        self.touch_activity()
        self.abandon_prompt_selection()
        # A fresh install has no API key yet, so recording is blocked and the settings window
        # is opened so the user can add a key first.
        if not transcription_configured(self._config):
            self.push_to_talk_active = False
            self._notifier.show(self._ctx.tr("recording_no_api_keys"), BALLOON_INFO_MS)
            self._open_settings()
            return

        self._permissions.warn('microphone')
        self._warmup.schedule(activate=True)
        self.is_recording = True
        self.is_paused = False
        self._take += 1
        self._feedback_pending = True
        self._wave_levels.clear()
        self._mac_device_name = ""
        self.state_changed.emit(True)

        if self.keep_mic_hot():
            self.hot_mic.start_recording()
            self.start_background_capture()
        elif macos_recorder.available():
            try:
                self.mac_recorder.start(self._ctx.recordings.new_path())
            except Exception as e:
                self.is_recording = False
                self.state_changed.emit(False)
                logging.error(f"macOS: could not start native audio recorder: {e}")
                self._notifier.show(self._ctx.tr("no_microphone_signal_message"), BALLOON_INFO_MS)
                return
        else:
            self.on_demand.start_recording()

        self._capture_selection_context()
        self._sounds.play('sound_start.wav')
        if self.mac_recorder.active:
            # AVAudioRecorder.record() only returns success once the input is recording.
            self._on_recording_audio(self._take)
        else:
            take = self._take
            QTimer.singleShot(MIC_SIGNAL_GRACE_MS, lambda: self._on_no_recording_audio_yet(take))
        logging.info("Recording started.")

    def _on_recording_audio(self, take: int) -> None:
        """The microphone delivers audio for this take: show the recording feedback now."""
        if take != self._take or not self.is_recording or not self._feedback_pending:
            return
        self._feedback_pending = False
        self._show_feedback()

    def _on_no_recording_audio_yet(self, take: int) -> None:
        """No audio yet after the grace period: show the plain recording balloon meanwhile."""
        if take != self._take or not self.is_recording or not self._feedback_pending:
            return
        self._notifier.show(self._ctx.tr("recording_running_message"), BALLOON_PERSISTENT_MS)

    def restart(self) -> None:
        """Start the running recording over: discard the take so far and keep recording.

        The palette, its prompt choice and push-to-talk stay as they are; only the elapsed
        time counts from zero again.
        """
        if not self.is_recording:
            return
        self.touch_activity()
        if self.mac_recorder.active:
            self.mac_recorder.stop(discard=True)
            try:
                self.mac_recorder.start(self._ctx.recordings.new_path())
            except Exception as e:
                logging.error(f"macOS: could not restart native audio recorder: {e}")
                self.is_recording = False
                self.push_to_talk_active = False
                self.abandon_prompt_selection()
                self.state_changed.emit(False)
                self._notifier.show(self._ctx.tr("no_microphone_signal_message"), BALLOON_INFO_MS)
                return
        else:
            self.active_capture().restart_recording()
        self.is_paused = False
        overlay = RecordingPromptOverlay._instance
        if overlay is not None:
            overlay.set_paused(False)
            overlay.restart_timer()
        logging.info("Recording restarted by user.")

    def toggle_pause(self) -> None:
        """Pause the running recording or continue it; a paused recording can still be stopped and sent."""
        if not self.is_recording:
            return
        self.touch_activity()
        paused = not self.is_paused
        if self.mac_recorder.active:
            if paused:
                self.mac_recorder.pause()
            else:
                self.mac_recorder.resume()
        else:
            self.active_capture().set_paused(paused)
        self.is_paused = paused
        overlay = RecordingPromptOverlay._instance
        if overlay is not None:
            overlay.set_paused(paused)
        # The status by the mouse that accompanies a palette by the status area tells the same state.
        if self._palette_status is not None and MouseFollowerTooltip._instance is self._palette_status:
            message = "recording_paused_message" if paused else "recording_running_message"
            self._notifier.show(self._ctx.tr(message), BALLOON_PERSISTENT_MS)
            self._palette_status = MouseFollowerTooltip._instance
        logging.info("Recording paused by user." if paused else "Recording resumed by user.")

    def _capture_selection_context(self) -> None:
        """Remember selected text as rephrasing context, after the mic is already hot."""
        self._pipeline.selection_context = ""
        if not instruction_entry(self._config.get("post_rephrasing_entries", []))["use_selection_context"]:
            return
        if is_console_foreground_window():
            logging.info("Skipping selection-context capture in console-like foreground window.")
            return
        context_text = self._output.get_selected_text()
        if context_text:
            self._pipeline.selection_context = context_text
            logging.info(f"Captured context for rephrasing: {redact_for_log(context_text)}")

    def _stop(self, timing: OperationTiming) -> None:
        """Stop capturing (retaining the in-flight block) and hand the audio to transcription."""
        timing.mark("stop_handled")
        self.touch_activity()
        self.is_recording = False
        self.is_paused = False
        self._feedback_pending = False
        # Stop the PyAudio capture first, so the block being read right now stays in the recording.
        collect: Optional[Callable[[], List[bytes]]] = None
        if not self.mac_recorder.active:
            collect = self.active_capture().stop_recording(timing)
        recording_prompt, use_auto_prompt = self.current_prompt, self.use_auto_prompt
        instruction_selected = self.current_prompt_is_instruction
        self._reset_prompt_choice()
        self._close_prompt_palette()
        self.push_to_talk_active = False
        self.state_changed.emit(False)
        self._notifier.show(self._ctx.tr("recording_stopped_message"), BALLOON_SHORT_MS)
        recorded_file_path = self.mac_recorder.stop() if collect is None else None
        frames = collect() if collect else []
        timing.mark("recording_stopped")
        logging.info("Recording stopped. Processing audio.")
        self._sounds.play('sound_end.wav')
        if recorded_file_path:
            self._process_recorded_file(recorded_file_path, recording_prompt, use_auto_prompt,
                                        instruction_selected, timing)
        else:
            self._finish(b"".join(frames), self.microphone.samplerate, self._ctx.recordings.new_path(),
                         recording_prompt, use_auto_prompt, instruction_selected, timing)

    def cancel(self) -> None:
        """Stops the current recording without processing it."""
        if not self.is_recording:
            return

        timing = OperationTiming("recording")
        timing.mark("stop_handled")

        logging.info("Recording canceled by user.")
        self.touch_activity()
        self.is_recording = False
        self.is_paused = False
        self._feedback_pending = False
        self.abandon_prompt_selection()
        self.push_to_talk_active = False

        if self.mac_recorder.active:
            self.mac_recorder.stop(discard=True)
        else:
            self.active_capture().cancel_recording()
        timing.mark("recording_stopped")
        timing.finish("cancelled")

        self.state_changed.emit(False)
        self._notifier.show(self._ctx.tr("recording_canceled_message"), BALLOON_SHORT_MS)
        self._sounds.play('sound_end.wav')

    # --- Recorded audio ------------------------------------------------------------------
    def _min_recording_seconds(self) -> float:
        """Configured minimum length; shorter takes are an accidental double-tap, not speech."""
        try:
            return float(self._config.get("min_recording_seconds", 1.0))
        except (TypeError, ValueError):
            return 1.0

    def _gain_db(self) -> float:
        """Configured gain; a hand-edited non-numeric value must never lose the recording."""
        try:
            return float(self._config["gain_db"])
        except (ValueError, TypeError):
            logging.warning(f"Invalid gain_db value {self._config.get('gain_db')!r}; defaulting to 0.0 dB.")
            return 0.0

    def _finish(self, raw_audio: bytes, samplerate: int, filepath: str,
                transformation_prompt: Optional[str], use_auto_prompt: bool, instruction_selected: bool,
                timing: OperationTiming) -> None:
        """Validate captured PCM, write it as the retained WAV and start transcription."""
        timing.mark("audio_prepare_start")
        if not raw_audio:
            logging.warning("No audio data was recorded.")
            self._notifier.show(self._ctx.tr("no_audio_captured_message"), BALLOON_SHORT_MS)
            timing.finish("no_audio")
            return
        min_seconds = self._min_recording_seconds()
        duration = dsp.duration_seconds(raw_audio, samplerate or UPLOAD_SAMPLERATE)
        if min_seconds > 0 and duration < min_seconds:
            # Treat it as a cancel so the user can abort by tapping again, without a billable request.
            logging.info(f"Recording {duration:.2f}s is shorter than the {min_seconds:.2f}s minimum; "
                         "discarding without transcription.")
            self._notifier.show(self._ctx.tr("recording_too_short_message"), BALLOON_SHORT_MS)
            timing.finish("too_short")
            return
        if dsp.peak(raw_audio) == 0:
            logging.warning(f"Recorded audio contained only silence (device='{self._input_source_name()}', "
                            f"rate={samplerate} Hz).")
            self._notifier.show(self._ctx.tr("no_microphone_signal_message"), BALLOON_INFO_MS)
            timing.finish("silence")
            return
        audio_bytes, output_samplerate = dsp.prepare_for_upload(raw_audio, samplerate, UPLOAD_SAMPLERATE,
                                                                self._gain_db(),
                                                                bool(self._config.get("gain_normalize", False)))
        timing.mark("audio_prepare_end")
        try:
            with timing.span("file_write"):
                dsp.write_wav(filepath, audio_bytes, output_samplerate)
        except Exception as e:
            logging.error(f"Failed to write WAV file: {e}")
            self._notifier.show(self._ctx.tr("save_audio_failed_message"), BALLOON_ERROR_MS)
            timing.finish("save_failed")
            return
        logging.info(f"Recording saved to: {filepath}")
        self._ctx.prune_recordings()
        self._pipeline.start(filepath, transformation_prompt=transformation_prompt,
                             use_auto_prompt=use_auto_prompt, instruction_selected=instruction_selected,
                             timing=timing)
        # Menus re-check file availability once the request is already on its way.
        self._ctx.files_changed.emit()

    def _input_source_name(self) -> str:
        """Name of the device that produced the last recording, for diagnostics."""
        return macos_recorder.DEVICE_NAME if macos_recorder.available() else self.microphone.device_name

    def _process_recorded_file(self, filepath: str, transformation_prompt: Optional[str],
                               use_auto_prompt: bool, instruction_selected: bool, timing: OperationTiming) -> None:
        """Process a recorder-produced WAV file in place and start transcription."""
        try:
            with wave.open(filepath, 'rb') as wf:
                channels, sampwidth = wf.getnchannels(), wf.getsampwidth()
                samplerate = wf.getframerate()
                raw_audio = wf.readframes(wf.getnframes())
        except Exception as e:
            logging.error(f"Failed to read recorded audio file '{filepath}': {e}")
            self._notifier.show(self._ctx.tr("save_audio_failed_message"), BALLOON_ERROR_MS)
            timing.finish("audio_read_failed")
            return
        if channels != 1 or sampwidth != 2:
            logging.warning(f"macOS: unexpected native recorder format (channels={channels}, sampwidth={sampwidth}).")
        self._finish(raw_audio, samplerate, filepath, transformation_prompt, use_auto_prompt,
                     instruction_selected, timing)

    # --- Recording feedback and prompt palette ---------------------------------------------
    def _show_feedback(self) -> None:
        """Show either the ordinary recording balloon or the optional prompt palette."""
        prompts = recording_prompt_entries(self._config.get("post_rephrasing_entries", []))
        if not prompts:
            self._notifier.show(self._ctx.tr("recording_running_message"), BALLOON_PERSISTENT_MS)
            return
        if not rephrasing_configured(self._config):
            self._notifier.show(self._ctx.tr("recording_prompt_api_missing"), BALLOON_WARNING_MS)
            return
        self.show_prompt_palette(prompts)

    def show_prompt_palette(self, prompts: List[Dict[str, Any]]) -> None:
        """Show the fixed prompt selector for the current microphone recording."""
        self._notifier.hide()
        self._palette_status = None
        self._reset_prompt_choice()
        use_system_position = bool(self._config.get("recording_prompt_overlay_system_position", False))
        RecordingPromptOverlay(
            prompts=prompts,
            status_text=self._ctx.tr("recording_running_message"),
            none_text=self._ctx.tr("recording_prompt_none"),
            on_selection_changed=self._on_prompt_selected,
            use_system_position=use_system_position,
            system_anchor=self._palette_anchor(),
            dark=theme.resolve_dark(self._config.get("color_theme", "system"), QApplication.instance()),
            on_cancel=self.cancel,
            on_restart=self.restart,
            cancel_tooltip=self._ctx.tr("recording_cancel_tooltip"),
            restart_tooltip=self._ctx.tr("recording_restart_tooltip"),
            move_tooltip=self._ctx.tr("recording_move_tooltip"),
            on_pause=self.toggle_pause,
            paused_text=self._ctx.tr("recording_paused_message"),
            pause_tooltip=self._ctx.tr("recording_pause_tooltip"),
            resume_tooltip=self._ctx.tr("recording_resume_tooltip"),
            digit_keys=bool(self._config.get("recording_prompt_overlay_digit_keys", True)),
            level_feed=self.drain_wave_levels,
            device_name=self.input_device_display_name,
            # The native macOS recorder always uses the system default microphone.
            devices=None if self.mac_recorder.active else self._device_menu_entries,
            on_device_selected=None if self.mac_recorder.active else self.switch_input_device,
            device_tooltip=self._ctx.tr("recording_device_tooltip"),
        )
        if RecordingPromptOverlay._instance is not None:
            RecordingPromptOverlay._instance.set_configured_device(self._config.get("input_device_name", "") or "")
        # A system-positioned palette can be far away from the user's current work. Keep the
        # original mouse-following recording status as a lightweight local reminder. When the
        # palette itself is mouse-relative, its own status label already provides that feedback.
        if use_system_position and (is_WINDOWS or is_MACOS):
            self._notifier.show(self._ctx.tr("recording_running_message"), BALLOON_PERSISTENT_MS)
            self._palette_status = MouseFollowerTooltip._instance

    def _on_prompt_selected(self, prompt: Optional[Dict[str, Any]]) -> None:
        """Remember the clicked palette choice until this recording is stopped.

        A prompt click is an explicit choice (it beats LivePrompt triggers); "None" (``None``)
        delivers the raw transcription by skipping the automatic prompt.
        """
        self.current_prompt = prompt["text"] if prompt else None
        self.current_prompt_is_instruction = bool(prompt and prompt.get("kind") == INSTRUCTION)
        self.use_auto_prompt = prompt is not None
        if prompt:
            self._warmup.schedule(activate=True)

    def _reset_prompt_choice(self) -> None:
        """Forget the palette choice: no explicit prompt, automatic prompt allowed."""
        self.current_prompt, self.current_prompt_is_instruction, self.use_auto_prompt = None, False, True

    def rephrasing_expected(self) -> bool:
        """Whether this recording will be rephrased by a palette or automatic prompt."""
        if self.current_prompt:
            return True
        return self.use_auto_prompt and auto_apply_prompt(self._config.get("post_rephrasing_entries", [])) is not None

    def _close_prompt_palette(self) -> None:
        """Close the recording selector without changing the already selected prompt."""
        RecordingPromptOverlay.close_current()
        mouse_status = self._palette_status
        self._palette_status = None
        # Another operation may already have replaced the recording status with its own tooltip.
        # In that case the replacement belongs to that operation and must remain visible.
        if mouse_status is not None and MouseFollowerTooltip._instance is mouse_status:
            mouse_status.close()

    def abandon_prompt_selection(self) -> None:
        """Close the selector and discard its choice when no request will use it."""
        self._reset_prompt_choice()
        self._close_prompt_palette()
