"""Shared constants and request/response types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import numpy as np


class SttLibrary(str, Enum):
    """Speech-to-text library."""

    AUTO = "auto"
    FASTER_WHISPER = "faster-whisper"
    OPENAI = "openai"


class Task(str, Enum):
    """Supported transcription tasks."""

    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"


AUTO_LANGUAGE = "auto"
AUTO_MODEL = "auto"
TARGET_SAMPLE_RATE = 16000
SUPPORTED_RESPONSE_FORMATS = {"json", "text"}
UNSUPPORTED_RESPONSE_FORMATS = {"verbose_json", "srt", "vtt", "diarized_json"}


@dataclass(frozen=True)
class AudioInput:
    """Uploaded or synthesized audio payload."""

    data: bytes
    filename: str
    content_type: Optional[str] = None


@dataclass(frozen=True)
class NormalizedAudio:
    """Mono float32 PCM audio at the target sample rate."""

    samples: np.ndarray
    sample_rate: int = TARGET_SAMPLE_RATE


@dataclass(frozen=True)
class TranscriptionRequest:
    """Protocol-agnostic transcription request."""

    audio: AudioInput
    model: Optional[str] = None
    language: Optional[str] = None
    task: Task = Task.TRANSCRIBE
    initial_prompt: Optional[str] = None
    speaker_enabled: bool = False


@dataclass(frozen=True)
class ModelSelection:
    """Model alias resolution result."""

    model: Optional[str]
    speaker_enabled: bool = False


@dataclass(frozen=True)
class TranscriptionResult:
    """Protocol-agnostic transcription result."""

    text: str
    model: str
    speaker: Optional[str] = None
    speaker_score: Optional[float] = None


AudioLike = Union[str, Path, np.ndarray]


class Transcriber(ABC):
    """Base class for transcribers."""

    @abstractmethod
    def transcribe(
        self,
        audio: AudioLike,
        language: Optional[str],
        task: Task = Task.TRANSCRIBE,
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> str:
        """Transcribe or translate audio input."""
