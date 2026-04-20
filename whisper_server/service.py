"""Shared request orchestration for Wyoming and HTTP transports."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import replace
from typing import Optional

from openai import AsyncOpenAI

from .audio import normalize_audio, temporary_audio_file
from .const import ModelSelection, Task, TranscriptionRequest, TranscriptionResult
from .key_manager import KeyManager
from .models import ModelLoader
from .speaker_recognition import SpeakerRecognizer
from .storage import StorageManager

_LOGGER = logging.getLogger(__name__)

DEFAULT_LLM_PROMPT = (
    "You are a helpful assistant that corrects speech-to-text transcriptions. "
    "Fix grammar, punctuation, and obvious misspellings, but maintain the original meaning and tone. "
    "Only return the corrected text."
)


class InvalidTranscriptionRequest(ValueError):
    """Raised when request input cannot be processed."""


class SpeechService:
    """Coordinates audio normalization, model selection, STT, and speaker tagging."""

    def __init__(
        self,
        loader: ModelLoader,
        speaker_recognizer: Optional[SpeakerRecognizer] = None,
        key_manager: Optional[KeyManager] = None,
        storage_manager: Optional[StorageManager] = None,
    ) -> None:
        self._loader = loader
        self._speaker_recognizer = speaker_recognizer
        self._key_manager = key_manager
        self._storage_manager = storage_manager

        llm_base_url = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
        llm_api_key = os.environ.get("LLM_API_KEY", "lm-studio")
        self._llm_model = os.environ.get("LLM_MODEL", "local-model")
        self._llm_client = AsyncOpenAI(base_url=llm_base_url, api_key=llm_api_key)

    @property
    def provider(self) -> str:
        return self._loader.provider

    async def warmup(
        self, model: Optional[str] = None, language: Optional[str] = None
    ) -> str:
        """Load the default or overridden transcriber into cache."""
        selection = parse_model_alias(model)
        resolved_model, _transcriber = await self._loader.load_transcriber(
            language=language, requested_model=selection.model
        )
        return resolved_model

    async def transcribe(
        self, request: TranscriptionRequest, api_key: Optional[str] = None
    ) -> TranscriptionResult:
        """Process a request through the shared service pipeline."""
        request = self._resolve_request(request)
        resolved_model, transcriber = await self._loader.load_transcriber(
            language=request.language, requested_model=request.model
        )

        normalized_audio = None
        if (self.provider != "openai") or request.speaker_enabled or self._storage_manager:
            try:
                normalized_audio = await asyncio.to_thread(normalize_audio, request.audio)
            except Exception as err:
                raise InvalidTranscriptionRequest(f"failed to decode audio: {err}") from err

        if self.provider == "openai":
            text = await asyncio.to_thread(
                self._transcribe_openai,
                transcriber,
                request,
            )
        else:
            assert normalized_audio is not None
            text = await asyncio.to_thread(
                transcriber.transcribe,
                normalized_audio.samples,
                request.language,
                request.task,
                self._loader.beam_size,
                request.initial_prompt,
                request.vad_filter,
            )

        raw_text = text
        llm_text = None
        if text and request.llm_correct:
            llm_text = await self._correct_with_llm(text, request.llm_prompt)
            text = llm_text

        # Post-transcription tasks (logging/storage)
        if api_key and self._key_manager:
            await self._key_manager.log_usage(api_key)

        if self._storage_manager:
            key_config = self._key_manager.get_key_config(api_key) if api_key and self._key_manager else None
            key_name = key_config.name if key_config else "anonymous"
            
            # Use original uploaded data if available, else normalized
            audio_bytes = request.audio.data
            await self._storage_manager.save_request(
                audio_bytes,
                raw_text,
                llm_text,
                key_name,
            )

        speaker = None
        speaker_score = None
        if text and request.speaker_enabled and self._speaker_recognizer:
            assert normalized_audio is not None
            speaker_result = await asyncio.to_thread(
                self._speaker_recognizer.identify, normalized_audio.samples
            )
            if speaker_result:
                speaker, speaker_score = speaker_result
                text = f"[{speaker}] {text}"

        return TranscriptionResult(
            text=text,
            model=resolved_model,
            speaker=speaker,
            speaker_score=speaker_score,
        )

    def _resolve_request(self, request: TranscriptionRequest) -> TranscriptionRequest:
        selection = parse_model_alias(request.model)
        prompt = request.initial_prompt
        if prompt is None:
            prompt = self._loader.initial_prompt

        language = request.language or self._loader.preferred_language
        return replace(
            request,
            model=selection.model,
            language=language,
            initial_prompt=prompt,
            speaker_enabled=request.speaker_enabled or selection.speaker_enabled,
        )

    @staticmethod
    def _transcribe_openai(transcriber, request: TranscriptionRequest) -> str:
        with temporary_audio_file(request.audio) as audio_path:
            return transcriber.transcribe(
                audio_path,
                request.language,
                request.task,
                initial_prompt=request.initial_prompt,
                vad_filter=request.vad_filter,
            )

    async def _correct_with_llm(self, text: str, custom_prompt: Optional[str]) -> str:
        """Use an LLM to refine the transcription text."""
        system_prompt = custom_prompt or DEFAULT_LLM_PROMPT
        try:
            response = await self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
            )
            corrected_text = response.choices[0].message.content
            return corrected_text.strip() if corrected_text else text
        except Exception as err:
            _LOGGER.warning("LLM correction failed: %s", err)
            return text


def parse_model_alias(model: Optional[str]) -> ModelSelection:
    """Parse the `speaker:<model>` alias."""
    if model is None:
        return ModelSelection(model=None, speaker_enabled=False)

    if model.startswith("speaker:"):
        resolved_model = model[len("speaker:") :].strip()
        if not resolved_model:
            raise InvalidTranscriptionRequest("speaker: model alias requires a model name")
        return ModelSelection(model=resolved_model, speaker_enabled=True)

    return ModelSelection(model=model, speaker_enabled=False)
