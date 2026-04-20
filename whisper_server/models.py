"""Model selection, loading, and caching."""

from __future__ import annotations

import asyncio
import gc
import logging
import platform
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .const import SttLibrary, Transcriber
from .faster_whisper_handler import FasterWhisperTranscriber

_LOGGER = logging.getLogger(__name__)

TRANSCRIBER_KEY = Tuple[SttLibrary, str]

# Strictly Official Systran Models + Verified Turbo
PREDEFINED_MODELS: Dict[str, str] = {
    "tiny": "Systran/faster-whisper-tiny",
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "base": "Systran/faster-whisper-base",
    "base.en": "Systran/faster-whisper-base.en",
    "small": "Systran/faster-whisper-small",
    "small.en": "Systran/faster-whisper-small.en",
    "medium": "Systran/faster-whisper-medium",
    "medium.en": "Systran/faster-whisper-medium.en",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large-v3",
    "turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
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
        self._last_used: Dict[TRANSCRIBER_KEY, float] = {}
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
                self._last_used[key] = time.time()
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
            self._last_used[key] = time.time()

        return model, transcriber

    def get_models_info(self) -> List[Dict[str, Any]]:
        """Get information about available, downloaded, and loaded models."""
        models_dir = self.download_dir / "models"
        downloaded_models = set()
        if models_dir.exists():
            # faster-whisper/ctranslate2 models are usually directories containing model.bin
            for p in models_dir.iterdir():
                if p.is_dir():
                    # Handle both "Systran/faster-whisper-tiny" and flat "tiny"
                    # In our download_dir/models, it might be nested or flat.
                    # faster-whisper uses huggingface_hub which creates nested structures usually,
                    # but sometimes they are flat if manually downloaded.
                    downloaded_models.add(p.name)
                    # Check for nested (org/model)
                    for sub in p.iterdir():
                        if sub.is_dir():
                            downloaded_models.add(f"{p.name}/{sub.name}")

        info = []
        for alias, model_id in PREDEFINED_MODELS.items():
            # A model is "downloaded" if its full ID or its last part is in downloaded_models
            is_downloaded = (
                model_id in downloaded_models 
                or model_id.split("/")[-1] in downloaded_models
                or (models_dir / model_id).exists()
            )
            
            # Check if loaded in memory
            is_ready = False
            for (lib, m), _ in self._transcriber.items():
                if m == model_id:
                    is_ready = True
                    break

            info.append({
                "id": alias,
                "model_id": model_id,
                "ready": is_ready,
                "downloaded": is_downloaded,
                "available": True
            })

        # Add currently loaded models that might not be in PREDEFINED_MODELS
        loaded_ids = {m for (_, m) in self._transcriber.keys()}
        predefined_ids = set(PREDEFINED_MODELS.values())
        for model_id in loaded_ids:
            if model_id not in predefined_ids:
                info.append({
                    "id": model_id,
                    "model_id": model_id,
                    "ready": True,
                    "downloaded": True,
                    "available": True
                })

        return info

    async def unload_idle_models(self, ttl_seconds: float) -> None:
        """Unload models that haven't been used for ttl_seconds."""
        now = time.time()
        to_unload = []

        for key, last_used in list(self._last_used.items()):
            if now - last_used > ttl_seconds:
                to_unload.append(key)

        for key in to_unload:
            async with self._transcriber_lock[key]:
                if key in self._transcriber:
                    _LOGGER.info("Unloading idle model: %s", key[1])
                    del self._transcriber[key]
                    del self._last_used[key]
                    
                    # Force garbage collection
                    gc.collect()
                    
                    # Try to clear CUDA cache if torch is available
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass

    async def unload_model(self, model_id: str) -> bool:
        """Manually unload a model by its ID or alias."""
        model_id = normalize_model_name(model_id)
        unloaded = False
        
        for key in list(self._transcriber.keys()):
            if key[1] == model_id:
                async with self._transcriber_lock[key]:
                    if key in self._transcriber:
                        _LOGGER.info("Manually unloading model: %s", model_id)
                        del self._transcriber[key]
                        if key in self._last_used:
                            del self._last_used[key]
                        unloaded = True

        if unloaded:
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
                
        return unloaded


def guess_model(stt_library: SttLibrary, language: Optional[str], is_arm: bool) -> str:
    """Automatically guess a default local STT model."""
    del stt_library, language
    if is_arm:
        return "Systran/faster-whisper-tiny"

    return "Systran/faster-whisper-base"


def normalize_model_name(model: str) -> str:
    """Normalize predefined aliases to full HuggingFace model IDs."""
    predefined = PREDEFINED_MODELS.get(model)
    if predefined:
        return predefined

    return model


def is_valid_model_name(model: str) -> bool:
    """Check if a model name is valid."""
    if not model:
        return False

    if model in PREDEFINED_MODELS:
        return True

    if "/" in model:
        return True

    if model.startswith("/") or model.startswith(".") or model.startswith("~"):
        return True

    return False
