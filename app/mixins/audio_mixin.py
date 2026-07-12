"""AudioMixin — recording lifecycle, input-device selection, capture loop and DSP."""
from __future__ import annotations

import logging
import os
import threading
import time
import wave
from typing import Any, Dict, List, Optional

import pyaudio

from app.audio.device_selector import InputDeviceSelector
from app.core import dsp
from app.core.env import is_MACOS, is_WINDOWS, open_with_default_app
from app.core.frameworks import NSURL, AVAudioRecorder
from app.core.redaction import redact_for_log

# The "recording…" balloon has no natural timeout — it stays until the stop/cancel path
# replaces it. Effectively "forever"; the tooltip's own safety cap is the backstop.
_RECORDING_BALLOON_TIMEOUT_MS = 99_999_999


class AudioMixin:
    """Recording lifecycle, input-device selection, capture loop, DSP and native macOS recorder."""

    def _get_pcm_peak(self, pcm_chunk: bytes) -> int:
        """Return the absolute peak of a 16-bit PCM chunk (delegates to core.dsp)."""
        return dsp.peak(pcm_chunk)

    def _is_recording_too_short(self, raw_audio: bytes, samplerate: int) -> bool:
        """Return True if the captured audio is shorter than the configured minimum.

        Ultra-short recordings are almost always an accidental double-tap of the hotkey; we
        treat them as a cancel so the user can immediately abort by tapping again, instead of
        firing off a pointless (and billable) transcription request.
        """
        try:
            min_seconds = float(self.config.get("min_recording_seconds", 1.0))
        except (TypeError, ValueError):
            min_seconds = 1.0
        if min_seconds <= 0:
            return False
        duration = dsp.duration_seconds(raw_audio, samplerate or self.samplerate)
        if duration < min_seconds:
            logging.info(
                f"Recording {duration:.2f}s is shorter than the {min_seconds:.2f}s minimum; "
                "discarding without transcription."
            )
            return True
        return False

    def _update_latest_audio_level(self, pcm_chunk: bytes) -> None:
        """Compute a lightweight peak meter value from a raw 16-bit PCM chunk."""
        max_sample = self._get_pcm_peak(pcm_chunk)
        normalized = max_sample / 32767.0 if max_sample else 0.0
        self.latest_audio_level = min(1.0, normalized * 1.8)

    def _start_background_audio_capture(self) -> None:
        """Start a background mic reader on Windows to avoid clipped leading audio."""
        if not is_WINDOWS or (self.audio_capture_thread and self.audio_capture_thread.is_alive()):
            return
        self._touch_transcription_activity()
        self.audio_capture_running = True
        self.audio_capture_thread = threading.Thread(target=self._audio_capture_loop, daemon=True)
        self.audio_capture_thread.start()

    def _stop_background_audio_capture(self) -> None:
        """Stop the Windows background mic reader cleanly."""
        self.audio_capture_running = False
        if self.audio_capture_thread and self.audio_capture_thread.is_alive():
            self.audio_capture_thread.join(timeout=1.5)
        self.audio_capture_thread = None
        self._close_input_stream()
        self._terminate_input_pyaudio_instance()

    def _audio_capture_loop(self) -> None:
        """Continuously read microphone audio and keep a short pre-roll buffer."""
        logging.info("Background audio capture thread started.")
        while self.audio_capture_running:
            if not self.is_recording:
                idle_seconds = self._get_windows_keep_mic_hot_idle_seconds()
                if idle_seconds > 0 and (time.monotonic() - self.last_transcription_activity_ts) >= idle_seconds:
                    logging.info("Stopping background audio capture due to Windows prewarm idle timeout.")
                    self.audio_capture_running = False
                    break
            try:
                stream = self._ensure_input_stream()
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                self._update_latest_audio_level(data)
                with self.audio_state_lock:
                    self.pre_record_buffer.append(data)
                    if self.is_recording:
                        self.recorded_frames.append(data)
            except Exception as e:
                logging.warning(f"Background audio capture error: {e}")
                self._close_input_stream()
                time.sleep(0.2)
        self._close_input_stream()
        self.audio_capture_thread = None
        logging.info("Background audio capture thread finished.")

    def play_sound(self, filename: str) -> None:
        """Low-latency playback of a preloaded short WAV (delegates to SoundPlayer)."""
        self.sound_player.play(filename)

    def toggle_recording(self) -> None:
        """Toggles the audio recording state."""
        if self.is_recording:
            self._touch_transcription_activity()
            self.is_recording = False
            self.push_to_talk_active = False
            self.cancel_action.setEnabled(False)  # Disable cancel while idle
            self._set_idle_tray_icon()
            self.show_tray_balloon(self.translator.tr("recording_stopped_message"), 2000)
            recorded_file_path: Optional[str] = None
            if is_MACOS and self.macos_audio_recorder:
                recorded_file_path = self._stop_macos_native_recording()
            elif (not self._use_windows_keep_mic_hot()) and self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join()
            logging.info("Recording stopped. Processing audio.")
            self.play_sound('sound_end.wav')
            if recorded_file_path:
                self._process_recorded_file(recorded_file_path)
            else:
                self.process_recording()
        else:
            self._touch_transcription_activity()
            # Check if API settings are complete before starting recording.
            # A fresh install has no API key yet, so recording is blocked and the
            # settings window is opened so the user can add a key first.
            api_url = self.api_endpoint_input.text().strip()
            api_key = self.api_key_input.text().strip()
            if not api_url or not api_key:
                self.push_to_talk_active = False
                self.show_tray_balloon(self.translator.tr("recording_no_api_keys"), 2500)
                self.show_settings_window()
                return

            self._check_and_warn_macos_permissions('microphone')

            self.is_recording = True
            self.cancel_action.setEnabled(True)  # Enable cancel while recording
            with self.audio_state_lock:
                self.recorded_frames = list(self.pre_record_buffer) if self._use_windows_keep_mic_hot() else []
            self._set_recording_tray_icon_active()

            if self._use_windows_keep_mic_hot():
                self._start_background_audio_capture()
            elif self._can_use_macos_native_recorder():
                try:
                    self._start_macos_native_recording()
                except Exception as e:
                    self.is_recording = False
                    self.cancel_action.setEnabled(False)
                    self._set_idle_tray_icon()
                    logging.error(f"macOS extension: Could not start native audio recorder: {e}")
                    self.show_tray_balloon(self.translator.tr("no_microphone_signal_message"), 2500)
                    return
            else:
                self.recording_thread = threading.Thread(target=self.record_audio, daemon=True)
                self.recording_thread.start()

            # Reset context for the new operation after the mic is already hot.
            self.current_transcription_context = ""
            if self.config["rephrase_use_selection_context"]:
                if is_WINDOWS and self._is_console_like_foreground_window():
                    logging.info("Skipping selection-context capture in console-like foreground window.")
                else:
                    context_text = self.get_selected_text()
                    if context_text:
                        self.current_transcription_context = context_text
                        logging.info(f"Captured context for rephrasing: {redact_for_log(context_text)}")

            self.play_sound('sound_start.wav')
            self.show_tray_balloon(self.translator.tr("recording_running_message"), _RECORDING_BALLOON_TIMEOUT_MS)
            logging.info("Recording started.")

    def cancel_recording(self) -> None:
        """Stops the current recording without processing it."""
        if not self.is_recording:
            return

        logging.info("Recording canceled by user.")
        self._touch_transcription_activity()
        self.is_recording = False
        self.push_to_talk_active = False
        self.cancel_action.setEnabled(False)  # Hide the action again

        # Wait for the recording thread to finish cleanly
        if is_MACOS and self.macos_audio_recorder:
            self._stop_macos_native_recording(discard=True)
        elif (not self._use_windows_keep_mic_hot()) and self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join()

        # Reset UI and provide feedback
        self._set_idle_tray_icon()
        self.show_tray_balloon(self.translator.tr("recording_canceled_message"), 2000)
        self.play_sound('sound_end.wav')

    def record_audio(self) -> None:
        """Records audio using PyAudio (16-bit mono)."""
        logging.info("Audio recording thread started (PyAudio).")
        try:
            stream = self._ensure_input_stream()
        except Exception as e:
            logging.error(f"Could not open audio input stream: {e}")
            self.is_recording = False
            return
        if self.current_input_device_name:
            logging.info(
                "Recording from input device "
                f"'{self.current_input_device_name}' "
                f"(index={self.current_input_device_index}, rate={self.current_input_samplerate} Hz)."
            )
        while self.is_recording:
            try:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                self._update_latest_audio_level(data)
                self.recorded_frames.append(data)
            except Exception as e:
                logging.error(f"Error while reading audio stream: {e}")
                break
        self._close_input_stream()
        logging.info("Audio recording thread finished.")

    def _prepare_pcm_audio_for_upload(self, raw_audio: bytes, source_samplerate: int) -> tuple[bytes, int]:
        """Normalize PCM audio to the app target sample rate + gain before upload."""
        output_samplerate = int(source_samplerate or self.samplerate)

        # macOS extension: native capture stays at 44.1 kHz for reliability, but we
        # downsample to the app target rate before writing/upload to keep files small.
        if output_samplerate != self.samplerate:
            try:
                raw_audio = dsp.resample(raw_audio, output_samplerate, self.samplerate)
                logging.info(
                    "macOS extension: Resampled recording from "
                    f"{output_samplerate} Hz to {self.samplerate} Hz before upload."
                )
                output_samplerate = self.samplerate
            except Exception as e:
                logging.warning(
                    f"Could not resample recording from {output_samplerate} Hz to {self.samplerate} Hz: {e}"
                )

        try:
            gain_db = float(self.config["gain_db"])
        except (ValueError, TypeError):
            # A hand-edited config may hold a non-numeric gain_db; never let that
            # abort processing (which would lose the recording). Mirror the
            # save_settings guard and fall back to no gain.
            logging.warning(
                f"Invalid gain_db value {self.config.get('gain_db')!r}; defaulting to 0.0 dB."
            )
            gain_db = 0.0

        audio_bytes = dsp.apply_gain(raw_audio, gain_db)
        if gain_db > 0:
            logging.info(f"Applied +{gain_db} dB gain to recording.")

        return audio_bytes, output_samplerate

    def process_recording(self) -> None:
        """Processes the recorded audio, saves it to a file, and starts transcription."""
        with self.audio_state_lock:
            recorded_frames = list(self.recorded_frames)

        if not recorded_frames:
            logging.warning("No audio data was recorded.")
            self.show_tray_balloon(self.translator.tr("no_audio_captured_message"), 2000)
            return
        raw_audio = b''.join(recorded_frames)
        if self._is_recording_too_short(raw_audio, self.current_input_samplerate):
            self.show_tray_balloon(self.translator.tr("recording_too_short_message"), 2000)
            return
        filepath: str = self.recordings.new_path()
        if self._get_pcm_peak(raw_audio) == 0:
            if is_MACOS:
                logging.warning(
                    "macOS extension: Recorded audio contained only silence "
                    f"(device='{self.current_input_device_name}', index={self.current_input_device_index}, "
                    f"rate={self.current_input_samplerate} Hz)."
                )
            else:
                logging.warning("Recorded audio contained only silence.")
            self.show_tray_balloon(self.translator.tr("no_microphone_signal_message"), 2500)
            return
        audio_bytes, output_samplerate = self._prepare_pcm_audio_for_upload(
            raw_audio,
            self.current_input_samplerate,
        )
        try:
            dsp.write_wav(filepath, audio_bytes, output_samplerate)
            logging.info(f"Recording saved to: {filepath}")
            # Enable the play action in the tray menu now that a file exists
            if hasattr(self, 'play_action'):
                self.play_action.setEnabled(True)
            self.update_play_last_recording_action()
        except Exception as e:
            logging.error(f"Failed to write WAV file: {e}")
            self.show_tray_balloon("Failed to save audio.", 3000)
            return
        self.keep_only_latest_recording()
        self.start_transcription_worker(filepath)

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
            self.show_tray_balloon(self.translator.tr("no_recording_found_message"), 2000)
            return
        try:
            open_with_default_app(latest)
        except Exception as e:
            self.show_tray_balloon(self.translator.tr("could_not_play_file_message", error=e), 2000)

    def _use_windows_keep_mic_hot(self) -> bool:
        """Return whether the Windows background microphone prewarm mode is enabled."""
        return is_WINDOWS and self.config.get("windows_keep_mic_hot", True)

    def _touch_transcription_activity(self) -> None:
        """Update the timestamp used for Windows microphone prewarm idle timeout."""
        self.last_transcription_activity_ts = time.monotonic()

    def _get_windows_keep_mic_hot_idle_seconds(self) -> float:
        """Return the configured Windows prewarm idle timeout in seconds."""
        minutes = max(0, int(self.config.get("windows_keep_mic_hot_idle_minutes", 15)))
        return float(minutes * 60)

    def _get_input_pyaudio_instance(self, refresh: bool = False) -> pyaudio.PyAudio:
        """Return the dedicated PyAudio instance used for microphone capture."""
        if refresh and self.input_pyaudio_instance:
            self._terminate_input_pyaudio_instance()
        if not self.input_pyaudio_instance:
            self.input_pyaudio_instance = pyaudio.PyAudio()
        return self.input_pyaudio_instance

    def _terminate_input_pyaudio_instance(self) -> None:
        """Tear down the dedicated microphone PyAudio instance."""
        if not self.input_pyaudio_instance:
            return
        try:
            self.input_pyaudio_instance.terminate()
        except Exception:
            pass
        self.input_pyaudio_instance = None

    def _get_input_device_selector(self) -> InputDeviceSelector:
        """Return the (lazily created) input-device enumeration helper."""
        if getattr(self, "_input_device_selector", None) is None:
            self._input_device_selector = InputDeviceSelector(self)
        return self._input_device_selector

    def selectable_input_devices(self) -> List[Dict[str, Any]]:
        """Input devices on the platform's default host API, for the settings dropdown."""
        return self._get_input_device_selector().selectable_input_devices()

    def apply_input_device_selection(self) -> None:
        """Reopen the capture stream on the newly selected input device (thread-safe).

        Closing a PortAudio stream while the background reader thread is inside stream.read()
        segfaults, so the reader is stopped (joined) BEFORE the stream is closed, then restarted.
        """
        if self.is_recording:
            # Don't yank the stream out from under an active recording.
            logging.info("Input device change deferred until the current recording stops.")
            return
        reader_running = bool(self.audio_capture_thread and self.audio_capture_thread.is_alive())
        if reader_running:
            self._stop_background_audio_capture()  # joins the reader, then closes the stream
        else:
            self._close_input_stream()
        if reader_running and self._use_windows_keep_mic_hot():
            self._start_background_audio_capture()
        logging.info(
            "Input device set to "
            f"{self.config.get('input_device_name') or 'System Default'!r}; capture stream reset."
        )

    def _get_preferred_input_stream_candidates(self, audio_instance: pyaudio.PyAudio) -> List[Dict[str, Any]]:
        """Build prioritized input stream candidates for the current platform."""
        return self._get_input_device_selector().preferred_input_stream_candidates(audio_instance)

    def _ensure_input_stream(self) -> Any:
        """Create and cache the PyAudio input stream used for low-latency recording."""
        if self.input_stream:
            return self.input_stream

        # macOS extension: Refresh the capture backend for each new recording.
        # This avoids stale CoreAudio device snapshots after app replacement,
        # permission changes or input-device switches.
        audio_instance = self._get_input_pyaudio_instance(refresh=is_MACOS)
        last_error: Optional[Exception] = None

        for candidate in self._get_preferred_input_stream_candidates(audio_instance):
            open_kwargs = {
                "format": pyaudio.paInt16,
                "channels": candidate["channels"],
                "rate": candidate["samplerate"],
                "input": True,
                "frames_per_buffer": self.chunk_size,
            }
            if candidate["index"] is not None:
                open_kwargs["input_device_index"] = candidate["index"]

            try:
                self.input_stream = audio_instance.open(**open_kwargs)
                self.current_input_samplerate = candidate["samplerate"]
                self.current_input_device_index = candidate["index"]
                self.current_input_device_name = candidate["name"]
                logging.info(
                    "Opened audio input stream on "
                    f"'{self.current_input_device_name}' "
                    f"(index={self.current_input_device_index}, rate={self.current_input_samplerate} Hz)."
                )
                self._notify_if_input_device_fallback(candidate)
                return self.input_stream
            except Exception as e:
                last_error = e
                logging.warning(
                    "Could not open audio input stream on "
                    f"'{candidate['name']}' (index={candidate['index']}, rate={candidate['samplerate']} Hz): {e}"
                )

        self.current_input_samplerate = self.samplerate
        self.current_input_device_index = None
        self.current_input_device_name = ""
        self._terminate_input_pyaudio_instance()
        if last_error:
            raise last_error
        raise RuntimeError("No usable audio input device could be opened.")

    def _notify_if_input_device_fallback(self, candidate: Dict[str, Any]) -> None:
        """Surface the silent fallback: a specific input device was configured, but it could not
        be opened, so recording fell back to the system default (candidate index is None).

        Deduplicated per configured device name so a repeatedly-reopened stream (prewarm reader,
        each recording) does not spam the notification. Runs on a capture thread — show_tray_balloon
        is thread-safe (it emits a queued signal).
        """
        configured = str(self.config.get("input_device_name", "") or "").strip()
        fell_back = configured and candidate.get("index") is None
        if fell_back:
            if getattr(self, "_last_input_fallback_notified", None) != configured:
                self._last_input_fallback_notified = configured
                logging.warning(
                    f"Configured input device {configured!r} could not be opened; "
                    "recording from the system default instead."
                )
                self.show_tray_balloon(
                    self.translator.tr("input_device_fallback_message", device=configured), 4000
                )
        else:
            # Configured device opened (or none configured): allow a future fallback to notify again.
            self._last_input_fallback_notified = None

    def _close_input_stream(self) -> None:
        """Close the cached PyAudio input stream."""
        if not self.input_stream:
            return
        try:
            self.input_stream.stop_stream()
        except Exception:
            pass
        try:
            self.input_stream.close()
        except Exception:
            pass
        self.input_stream = None
        if is_MACOS:
            self._terminate_input_pyaudio_instance()

    def _can_use_macos_native_recorder(self) -> bool:
        """Return whether the native macOS recorder backend is available."""
        return bool(is_MACOS and AVAudioRecorder and NSURL)

    def _start_macos_native_recording(self) -> None:
        """Start a native AVAudioRecorder capture on macOS."""
        if not self._can_use_macos_native_recorder():
            raise RuntimeError("Native macOS audio recorder is not available.")

        recording_path = self.recordings.new_path()
        if os.path.exists(recording_path):
            try:
                os.remove(recording_path)
            except Exception:
                pass

        settings = {
            "AVFormatIDKey": int.from_bytes(b"lpcm", "big"),
            "AVSampleRateKey": 44100.0,
            "AVNumberOfChannelsKey": 1,
            "AVLinearPCMBitDepthKey": 16,
            "AVLinearPCMIsBigEndianKey": False,
            "AVLinearPCMIsFloatKey": False,
        }

        recorder = AVAudioRecorder.alloc().initWithURL_settings_error_(
            NSURL.fileURLWithPath_(recording_path),
            settings,
            None,
        )
        recorder.setMeteringEnabled_(True)
        recorder.prepareToRecord()
        if not recorder.record():
            raise RuntimeError("AVAudioRecorder did not start recording.")

        self.macos_audio_recorder = recorder
        self.macos_recording_path = recording_path
        self.current_input_samplerate = int(settings["AVSampleRateKey"])
        self.current_input_device_index = None
        self.current_input_device_name = "macOS system microphone"
        logging.info(
            "macOS extension: Started native recorder capture "
            f"to '{self.macos_recording_path}' at {self.current_input_samplerate} Hz."
        )

    def _stop_macos_native_recording(self, discard: bool = False) -> Optional[str]:
        """Stop the native macOS recorder and optionally discard its output file."""
        recorder = self.macos_audio_recorder
        recording_path = self.macos_recording_path
        self.macos_audio_recorder = None
        self.macos_recording_path = None

        if recorder:
            try:
                recorder.stop()
            except Exception as e:
                logging.warning(f"macOS extension: Failed to stop native recorder cleanly: {e}")

        if not recording_path:
            return None

        time.sleep(0.08)

        if discard:
            if os.path.exists(recording_path):
                try:
                    os.remove(recording_path)
                except Exception as e:
                    logging.warning(f"macOS extension: Could not delete discarded recording '{recording_path}': {e}")
            return None

        return recording_path

    def _process_recorded_file(self, filepath: str) -> None:
        """Process a recorder-produced WAV file and start transcription."""
        try:
            with wave.open(filepath, 'rb') as wf:
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                samplerate = wf.getframerate()
                raw_audio = wf.readframes(wf.getnframes())
        except Exception as e:
            logging.error(f"Failed to read recorded audio file '{filepath}': {e}")
            self.show_tray_balloon("Failed to save audio.", 3000)
            return

        if not raw_audio:
            logging.warning("Recorded audio file was empty.")
            self.show_tray_balloon(self.translator.tr("no_audio_captured_message"), 2000)
            return

        if self._is_recording_too_short(raw_audio, samplerate):
            self.show_tray_balloon(self.translator.tr("recording_too_short_message"), 2000)
            return

        if channels != 1 or sampwidth != 2:
            logging.warning(
                "macOS extension: Unexpected native recorder format "
                f"(channels={channels}, sampwidth={sampwidth})."
            )

        self.current_input_samplerate = samplerate or self.current_input_samplerate
        if self._get_pcm_peak(raw_audio) == 0:
            logging.warning(
                "macOS extension: Native recorder captured only silence "
                f"(device='{self.current_input_device_name}', rate={self.current_input_samplerate} Hz)."
            )
            self.show_tray_balloon(self.translator.tr("no_microphone_signal_message"), 2500)
            return

        audio_bytes, output_samplerate = self._prepare_pcm_audio_for_upload(raw_audio, samplerate)

        try:
            dsp.write_wav(filepath, audio_bytes, output_samplerate)
            logging.info(f"Recording saved to: {filepath}")
            if hasattr(self, 'play_action'):
                self.play_action.setEnabled(True)
            self.update_play_last_recording_action()
        except Exception as e:
            logging.error(f"Failed to write WAV file: {e}")
            self.show_tray_balloon("Failed to save audio.", 3000)
            return

        self.keep_only_latest_recording()
        self.start_transcription_worker(filepath)
