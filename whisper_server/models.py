"""Model selection, loading, and caching."""

from __future__ import annotations

import asyncio
import logging
import platform
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from .const import SttLibrary, Transcriber
from .faster_whisper_handler import FasterWhisperTranscriber

_LOGGER = logging.getLogger(__name__)

TRANSCRIBER_KEY = Tuple[SttLibrary, str]

PREDEFINED_MODELS: Dict[str, str] = {
    # Official Systran Models (Recommended)
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large-v3",
    "turbo": "Systran/faster-whisper-large-v3-turbo",
    
    # Distil Variants (Fast & Efficient)
    "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
}


class ModelLoader:
    """Load transcribers for models and reuse them across requests."""

    def __init__(
        self,
        preferred_stt_library: SttLibrary,
        preferred_language: Optional[str],
        download_dir: Union[str, Path],
        local_files_only: bool,
        model: Optional[str],
        compute_type: str,
        device: str,
        beam_size: int,
        cpu_threads: int,
        initial_prompt: Optional[str],
        vad_parameters: Optional[Dict[str, Any]],
        provider: str = "local",
    ) -> None:
        self.preferred_stt_library = preferred_stt_library
        self.preferred_language = preferred_language
        self.download_dir = Path(download_dir)
        self.local_files_only = local_files_only
        self.model = model
        self.compute_type = compute_type
        self.device = device
        self.beam_size = beam_size
        self.cpu_threads = cpu_threads
        self.initial_prompt = initial_prompt
        self.vad_parameters = vad_parameters
        self.provider = provider

        self._transcriber: Dict[TRANSCRIBER_KEY, Transcriber] = {}
        self._transcriber_lock: Dict[TRANSCRIBER_KEY, asyncio.Lock] = defaultdict(
            asyncio.Lock
        )

    def resolve_model_name(
        self, requested_model: Optional[str], language: Optional[str] = None
    ) -> str:
        """Resolve the effective model for a request."""
        model = requested_model or self.model
        language = language or self.preferred_language

        if model is None:
            if self.provider == "openai":
                model = "gpt-4o-transcribe"
            else:
                machine = platform.machine().lower()
                is_arm = ("arm" in machine) or ("aarch" in machine)
                model = guess_model(self.preferred_stt_library, language, is_arm)

        return normalize_model_name(model)

    async def load_transcriber(
        self, language: Optional[str] = None, requested_model: Optional[str] = None
    ) -> tuple[str, Transcriber]:
        """Load or get a cached transcriber for the effective model."""
        language = language or self.preferred_language
        stt_library = self.preferred_stt_library
        if stt_library == SttLibrary.AUTO:
            stt_library = (
                SttLibrary.OPENAI
                if self.provider == "openai"
                else SttLibrary.FASTER_WHISPER
            )

        model = self.resolve_model_name(requested_model, language=language)
        _LOGGER.debug(
            "Selected stt-library '%s' with model '%s'", stt_library.value, model
        )

        key = (stt_library, model)
        async with self._transcriber_lock[key]:
            transcriber = self._transcriber.get(key)
            if transcriber is not None:
                return model, transcriber

            if self.provider == "openai":
                from .openai_transcriber import OpenAITranscriber

                transcriber = OpenAITranscriber(model_id=model)
            else:
                models_dir = self.download_dir / "models"
                models_dir.mkdir(parents=True, exist_ok=True)
                transcriber = FasterWhisperTranscriber(
                    model,
                    cache_dir=models_dir,
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=self.cpu_threads,
                    vad_parameters=self.vad_parameters,
                )

            self._transcriber[key] = transcriber

        return model, transcriber


def guess_model(stt_library: SttLibrary, language: Optional[str], is_arm: bool) -> str:
    """Automatically guess a default local STT model."""
    del stt_library, language
    if is_arm:
        return "Systran/faster-whisper-tiny"

    return "Systran/faster-whisper-base"


def normalize_model_name(model: str) -> str:
    """Normalize predefined aliases to full HuggingFace model IDs.

    Supports:
    - Official Systran models (tiny, base, small, medium, large, turbo)
    - Distil variants (distil-large-v2, distil-large-v3)
    - Full HuggingFace model IDs (e.g., Systran/faster-whisper-medium)
    """
    predefined = PREDEFINED_MODELS.get(model)
    if predefined:
        return predefined

    return model


def is_valid_model_name(model: str) -> bool:
    """Check if a model name is valid.

    Valid models are:
    - Predefined aliases (tiny, base, medium, large, etc.)
    - HuggingFace model IDs (contains "/")
    - Local paths (absolute or relative)
    """
    if not model:
        return False

    if model in PREDEFINED_MODELS:
        return True

    if "/" in model:
        return True

    if model.startswith("/") or model.startswith(".") or model.startswith("~"):
        return True

    return False
