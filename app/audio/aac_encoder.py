"""AAC/M4A encoding through PyAV's bundled FFmpeg libraries, identical on every platform."""
from __future__ import annotations

import logging
import wave
from functools import lru_cache

#: Bitrates offered in the settings; FFmpeg's AAC-LC encoder supports all of them at 48 kHz mono.
AAC_BITRATES = (48, 64, 96, 128, 160, 192)
#: Speech is resampled to 48 kHz so the higher bitrates are not clamped by a low sample rate.
OUTPUT_SAMPLERATE = 48000


@lru_cache(maxsize=1)
def available_aac_bitrates() -> tuple[int, ...]:
    """Return the offered bitrates if PyAV and its AAC encoder load; never raises."""
    try:
        import av

        av.codec.Codec("aac", "w")
    except Exception as error:  # Missing wheel, broken native libraries or a build without AAC.
        logging.warning("AAC encoder unavailable: %s", error)
        return ()
    return AAC_BITRATES


def encode_wav_to_aac(source: str, destination: str, bitrate_kbps: int) -> None:
    """Encode a prepared mono 16-bit PCM WAV to AAC-LC/M4A; the source is never modified.

    The WAV is streamed in 100 ms blocks, so memory stays bounded for long recordings.
    """
    if bitrate_kbps not in available_aac_bitrates():
        raise ValueError("AAC bitrate unavailable; choose WAV or another bitrate.")
    import av

    with wave.open(source, "rb") as recording:
        if recording.getnchannels() != 1 or recording.getsampwidth() != 2:
            raise ValueError("AAC encoding requires mono 16-bit PCM.")
        samplerate = recording.getframerate()
        block_frames = max(1, samplerate // 10)
        with av.open(destination, "w", format="mp4") as container:
            stream = container.add_stream("aac", rate=OUTPUT_SAMPLERATE, layout="mono")
            stream.bit_rate = bitrate_kbps * 1000
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=OUTPUT_SAMPLERATE)

            def encode(frame: av.AudioFrame | None) -> None:
                """Resample one block (None flushes) and mux every finished packet."""
                for resampled in resampler.resample(frame):
                    container.mux(stream.encode(resampled))

            while pcm := recording.readframes(block_frames):
                frame = av.AudioFrame(format="s16", layout="mono", samples=len(pcm) // 2)
                frame.planes[0].update(pcm)
                frame.sample_rate = samplerate
                encode(frame)
            encode(None)
            container.mux(stream.encode(None))
