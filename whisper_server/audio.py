"""Audio decode and normalization helpers."""

from __future__ import annotations

import io
import wave
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterator

import numpy as np
from faster_whisper.audio import decode_audio

from .const import AudioInput, NormalizedAudio, TARGET_SAMPLE_RATE


def normalize_audio(audio: AudioInput) -> NormalizedAudio:
    """Decode audio bytes into mono float32 PCM at 16 kHz."""
    samples = decode_audio(io.BytesIO(audio.data), sampling_rate=TARGET_SAMPLE_RATE)
    return NormalizedAudio(
        samples=np.asarray(samples, dtype=np.float32), sample_rate=TARGET_SAMPLE_RATE
    )


def normalize_audio_file(path: Path) -> NormalizedAudio:
    """Decode an audio file from disk into normalized PCM."""
    samples = decode_audio(str(path), sampling_rate=TARGET_SAMPLE_RATE)
    return NormalizedAudio(
        samples=np.asarray(samples, dtype=np.float32), sample_rate=TARGET_SAMPLE_RATE
    )


def wav_bytes_from_pcm16(pcm_bytes: bytes, sample_rate: int = TARGET_SAMPLE_RATE) -> bytes:
    """Wrap raw mono 16-bit PCM in a WAV container."""
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_bytes)
        return buffer.getvalue()


def wav_bytes_from_float32(
    samples: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE
) -> bytes:
    """Serialize normalized float32 samples as 16-bit PCM WAV."""
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2", copy=False).tobytes()
    return wav_bytes_from_pcm16(pcm, sample_rate=sample_rate)


@contextmanager
def temporary_audio_file(audio: AudioInput) -> Iterator[Path]:
    """Write uploaded audio to a temporary file for APIs that need a path."""
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(audio.data)
        tmp_path = Path(tmp_file.name)

    try:
        yield tmp_path
    finally:
        tmp_path.unlink(missing_ok=True)
