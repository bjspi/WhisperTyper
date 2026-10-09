"""AudioMixin — recording lifecycle: start/stop/cancel, capture strategy and result handling."""
from __future__ import annotations

import logging
import time
import wave
from typing import Any, Callable, Dict, List, Optional, Union

from app.audio import macos_recorder
from app.audio.capture import HotMicCapture, MicrophoneInput, OnDemandCapture
from app.core import dsp
from app.core.api_keys import rephrasing_configured, transcription_configured
from app.core.env import is_WINDOWS
from app.core.prompts import recording_prompt_entries
from app.core.redaction import redact_for_log
from app.core.timing import OperationTiming
from app.platform.system import open_with_default_app
from app.platform.windows import is_console_foreground_window
from app.ui.durations import (
    BALLOON_ERROR_MS,
    BALLOON_INFO_MS,
    BALLOON_PERSISTENT_MS,
    BALLOON_SHORT_MS,
    BALLOON_WARNING_MS,
)

#: Sample rate of uploaded recordings.
UPLOAD_SAMPLERATE = 16000
#: PyAudio frames per read.
CHUNK_SIZE = 1024


class AudioMixin:
    """Recording lifecycle on top of the capture strategies in :mod:`app.audio.capture`."""

    def _init_audio_capture(self) -> None:
        """Create the microphone, both PyAudio capture strategies and the macOS recorder."""
        self.is_recording = False
        self.latest_audio_level = 0.0
        self.last_transcription_activity_ts = time.monotonic()
        self._microphone = MicrophoneInput(
            lambda: self.config, UPLOAD_SAMPLERATE, CHUNK_SIZE, self._notify_input_device_fallback,
        )
        self._hot_mic = HotMicCapture(self._microphone, idle_expired=self._hot_mic_idle_expired,
                                      on_block=self._update_latest_audio_level)
        self._on_demand_capture = OnDemandCapture(self._microphone, on_block=self._update_latest_audio_level,
                                                  on_failure=self._on_capture_open_failed)
        self._mac_recorder = macos_recorder.MacRecorder()

    def _update_latest_audio_level(self, pcm_chunk: bytes) -> None:
        """Feed the tray meter from a captured block."""
        self.latest_audio_level = dsp.meter_level(pcm_chunk)

    def _on_capture_open_failed(self) -> None:
        """The per-recording stream could not be opened (capture thread): end the recording."""
        self.is_recording = False
        self.close_recording_prompt_overlay_signal.emit()

    def _notify_input_device_fallback(self, device: str) -> None:
        """Tell the user the configured device failed and the default is used (capture thread; thread-safe)."""
        self.show_tray_balloon(self.translator.tr("input_device_fallback_message", device=device), BALLOON_WARNING_MS)

    def _active_capture(self) -> Union[HotMicCapture, OnDemandCapture]:
        """The PyAudio capture strategy for the current settings."""
        return self._hot_mic if self._use_windows_keep_mic_hot() else self._on_demand_capture

    def _start_background_audio_capture(self) -> None:
        """Start the Windows background mic reader to avoid clipped leading audio."""
        if not is_WINDOWS or self._hot_mic.running:
            return
        self._touch_transcription_activity()
        self._hot_mic.start()

    def _stop_background_audio_capture(self) -> None:
        """Stop the Windows background mic reader and release the capture device."""
        self._hot_mic.stop()
        self._microphone.terminate()

    def _shutdown_audio_capture(self) -> None:
        """Release every capture resource at quit."""
        self._stop_background_audio_capture()
        if self._mac_recorder.active:
            self._mac_recorder.stop(discard=True)

    def play_sound(self, filename: str) -> None:
        """Low-latency playback of a preloaded short WAV (delegates to SoundPlayer)."""
        self.sound_player.play(filename)

    def toggle_recording(self, *, detected_ns: Optional[int] = None) -> None:
        """Toggles the audio recording state."""
        if self.is_recording:
            self._stop_recording(OperationTiming("recording", detected_ns))
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        """Start capturing, after checking that a transcription could be sent at all."""
        self._touch_transcription_activity()
        self._abandon_recording_prompt_selection()
        # A fresh install has no API key yet, so recording is blocked and the settings window
        # is opened so the user can add a key first.
        if not transcription_configured(self.config):
            self.push_to_talk_active = False
            self.show_tray_balloon(self.translator.tr("recording_no_api_keys"), BALLOON_INFO_MS)
            self.show_settings_window()
            return

        self._check_and_warn_macos_permissions('microphone')
        self._schedule_http_warmup(activate=True)
        self.cancel_action.setEnabled(True)  # Enable cancel while recording
        self.is_recording = True
        self._set_recording_tray_icon_active()

        if self._use_windows_keep_mic_hot():
            self._hot_mic.start_recording()
            self._start_background_audio_capture()
        elif macos_recorder.available():
            try:
                self._mac_recorder.start(self.recordings.new_path())
            except Exception as e:
                self.is_recording = False
                self.cancel_action.setEnabled(False)
                self._set_idle_tray_icon()
                logging.error(f"macOS: could not start native audio recorder: {e}")
                self.show_tray_balloon(self.translator.tr("no_microphone_signal_message"), BALLOON_INFO_MS)
                return
        else:
            self._on_demand_capture.start_recording()

        self._capture_selection_context()
        self.play_sound('sound_start.wav')
        self._show_recording_feedback()
        logging.info("Recording started.")

    def _capture_selection_context(self) -> None:
        """Remember selected text as rephrasing context, after the mic is already hot."""
        self.current_transcription_context = ""
        if not self.config["rephrase_use_selection_context"]:
            return
        if is_console_foreground_window():
            logging.info("Skipping selection-context capture in console-like foreground window.")
            return
        context_text = self.get_selected_text()
        if context_text:
            self.current_transcription_context = context_text
            logging.info(f"Captured context for rephrasing: {redact_for_log(context_text)}")

    def _stop_recording(self, timing: OperationTiming) -> None:
        """Stop capturing (retaining the in-flight block) and hand the audio to transcription."""
        timing.mark("stop_handled")
        self._touch_transcription_activity()
        self.is_recording = False
        # Stop the PyAudio capture first, so the block being read right now stays in the recording.
        collect: Optional[Callable[[], List[bytes]]] = None
        if not self._mac_recorder.active:
            collect = self._active_capture().stop_recording(timing)
        recording_prompt = self.current_recording_prompt
        self.current_recording_prompt = None
        self._close_recording_prompt_overlay()
        self.push_to_talk_active = False
        self.cancel_action.setEnabled(False)  # Disable cancel while idle
        self._set_idle_tray_icon()
        self.show_tray_balloon(self.translator.tr("recording_stopped_message"), BALLOON_SHORT_MS)
        recorded_file_path = self._mac_recorder.stop() if collect is None else None
        frames = collect() if collect else []
        timing.mark("recording_stopped")
        logging.info("Recording stopped. Processing audio.")
        self.play_sound('sound_end.wav')
        if recorded_file_path:
            self._process_recorded_file(recorded_file_path, recording_prompt, timing)
        else:
            self._finish_recording(b"".join(frames), self._microphone.samplerate, self.recordings.new_path(),
                                   recording_prompt, timing)

    def _show_recording_feedback(self) -> None:
        """Show either the ordinary recording balloon or the optional prompt palette."""
        prompts = recording_prompt_entries(self.config.get("post_rephrasing_entries", []))
        if not prompts:
            self.show_tray_balloon(self.translator.tr("recording_running_message"), BALLOON_PERSISTENT_MS)
            return
        if not rephrasing_configured(self.config):
            self.show_tray_balloon(self.translator.tr("recording_prompt_api_missing"), BALLOON_WARNING_MS)
            return
        self._show_recording_prompt_overlay(prompts)

    def cancel_recording(self) -> None:
        """Stops the current recording without processing it."""
        if not self.is_recording:
            return

        timing = OperationTiming("recording")
        timing.mark("stop_handled")

        logging.info("Recording canceled by user.")
        self._touch_transcription_activity()
        self.is_recording = False
        self._abandon_recording_prompt_selection()
        self.push_to_talk_active = False
        self.cancel_action.setEnabled(False)  # Hide the action again

        if self._mac_recorder.active:
            self._mac_recorder.stop(discard=True)
        else:
            self._active_capture().cancel_recording()
        timing.mark("recording_stopped")
        timing.finish("cancelled")

        # Reset UI and provide feedback
        self._set_idle_tray_icon()
        self.show_tray_balloon(self.translator.tr("recording_canceled_message"), BALLOON_SHORT_MS)
        self.play_sound('sound_end.wav')

    def _min_recording_seconds(self) -> float:
        """Configured minimum length; shorter takes are an accidental double-tap, not speech."""
        try:
            return float(self.config.get("min_recording_seconds", 1.0))
        except (TypeError, ValueError):
            return 1.0

    def _gain_db(self) -> float:
        """Configured gain; a hand-edited non-numeric value must never lose the recording."""
        try:
            return float(self.config["gain_db"])
        except (ValueError, TypeError):
            logging.warning(f"Invalid gain_db value {self.config.get('gain_db')!r}; defaulting to 0.0 dB.")
            return 0.0

    def _finish_recording(self, raw_audio: bytes, samplerate: int, filepath: str,
                          transformation_prompt: Optional[str], timing: OperationTiming) -> None:
        """Validate captured PCM, write it as the retained WAV and start transcription."""
        timing.mark("audio_prepare_start")
        if not raw_audio:
            logging.warning("No audio data was recorded.")
            self.show_tray_balloon(self.translator.tr("no_audio_captured_message"), BALLOON_SHORT_MS)
            timing.finish("no_audio")
            return
        min_seconds = self._min_recording_seconds()
        duration = dsp.duration_seconds(raw_audio, samplerate or UPLOAD_SAMPLERATE)
        if min_seconds > 0 and duration < min_seconds:
            # Treat it as a cancel so the user can abort by tapping again, without a billable request.
            logging.info(f"Recording {duration:.2f}s is shorter than the {min_seconds:.2f}s minimum; "
                         "discarding without transcription.")
            self.show_tray_balloon(self.translator.tr("recording_too_short_message"), BALLOON_SHORT_MS)
            timing.finish("too_short")
            return
        if dsp.peak(raw_audio) == 0:
            logging.warning(f"Recorded audio contained only silence (device='{self._input_source_name()}', "
                            f"rate={samplerate} Hz).")
            self.show_tray_balloon(self.translator.tr("no_microphone_signal_message"), BALLOON_INFO_MS)
            timing.finish("silence")
            return
        audio_bytes, output_samplerate = dsp.prepare_for_upload(raw_audio, samplerate, UPLOAD_SAMPLERATE,
                                                                self._gain_db())
        timing.mark("audio_prepare_end")
        try:
            with timing.span("file_write"):
                dsp.write_wav(filepath, audio_bytes, output_samplerate)
        except Exception as e:
            logging.error(f"Failed to write WAV file: {e}")
            self.show_tray_balloon(self.translator.tr("save_audio_failed_message"), BALLOON_ERROR_MS)
            timing.finish("save_failed")
            return
        logging.info(f"Recording saved to: {filepath}")
        self.update_play_last_recording_action()
        self.keep_only_latest_recording()
        self.start_transcription_worker(filepath, transformation_prompt=transformation_prompt, timing=timing)

    def _input_source_name(self) -> str:
        """Name of the device that produced the last recording, for diagnostics."""
        return macos_recorder.DEVICE_NAME if macos_recorder.available() else self._microphone.device_name

    def _process_recorded_file(self, filepath: str, transformation_prompt: Optional[str],
                               timing: OperationTiming) -> None:
        """Process a recorder-produced WAV file in place and start transcription."""
        try:
            with wave.open(filepath, 'rb') as wf:
                channels, sampwidth = wf.getnchannels(), wf.getsampwidth()
                samplerate = wf.getframerate()
                raw_audio = wf.readframes(wf.getnframes())
        except Exception as e:
            logging.error(f"Failed to read recorded audio file '{filepath}': {e}")
            self.show_tray_balloon(self.translator.tr("save_audio_failed_message"), BALLOON_ERROR_MS)
            timing.finish("audio_read_failed")
            return
        if channels != 1 or sampwidth != 2:
            logging.warning(f"macOS: unexpected native recorder format (channels={channels}, sampwidth={sampwidth}).")
        self._finish_recording(raw_audio, samplerate, filepath, transformation_prompt, timing)

    def cleanup_old_recordings(self) -> None:
        """Delete all old whispertyper_recording_*.wav files on startup."""
        self.recordings.cleanup_all()

    def keep_only_latest_recording(self) -> None:
        """Delete all but the newest recording file after a new recording is saved."""
        self.recordings.keep_only_latest()

    def play_latest_recording(self) -> None:
        """Open the latest recording in the system's default media player."""
        latest = self.recordings.latest()
        if not latest:
            self.show_tray_balloon(self.translator.tr("no_recording_found_message"), BALLOON_SHORT_MS)
            return
        try:
            open_with_default_app(latest)
        except Exception as e:
            self.show_tray_balloon(self.translator.tr("could_not_play_file_message", error=e), BALLOON_SHORT_MS)

    def _use_windows_keep_mic_hot(self) -> bool:
        """Return whether the Windows background microphone prewarm mode is enabled."""
        return is_WINDOWS and self.config.get("windows_keep_mic_hot", True)

    def _touch_transcription_activity(self) -> None:
        """Update the timestamp used for Windows microphone prewarm idle timeout."""
        self.last_transcription_activity_ts = time.monotonic()

    def _hot_mic_idle_expired(self) -> bool:
        """Whether the configured prewarm idle timeout elapsed since the last activity (0 = never)."""
        idle_seconds = max(0, int(self.config.get("windows_keep_mic_hot_idle_minutes", 15))) * 60
        return idle_seconds > 0 and time.monotonic() - self.last_transcription_activity_ts >= idle_seconds

    def selectable_input_devices(self) -> List[Dict[str, Any]]:
        """Input devices on the platform's default host API, for the settings dropdown."""
        return self._microphone.devices.selectable_input_devices()

    def apply_input_device_selection(self) -> None:
        """Reopen the capture stream on the newly selected input device (thread-safe).

        Closing a PortAudio stream while the background reader thread is inside stream.read()
        segfaults, so the reader is stopped (joined) BEFORE the stream is closed, then restarted.
        """
        if self.is_recording:
            # Don't yank the stream out from under an active recording.
            logging.info("Input device change deferred until the current recording stops.")
            return
        reader_running = self._hot_mic.running
        if reader_running:
            self._stop_background_audio_capture()  # joins the reader, then closes the stream
        else:
            self._microphone.close()
        if reader_running and self._use_windows_keep_mic_hot():
            self._start_background_audio_capture()
        logging.info(
            "Input device set to "
            f"{self.config.get('input_device_name') or 'System Default'!r}; capture stream reset."
        )
