"""Shared request orchestration for Wyoming and HTTP transports."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Optional

from .audio import normalize_audio, temporary_audio_file
from .const import ModelSelection, Task, TranscriptionRequest, TranscriptionResult
from .models import ModelLoader
from .speaker_recognition import SpeakerRecognizer


class InvalidTranscriptionRequest(ValueError):
    """Raised when request input cannot be processed."""


class SpeechService:
    """Coordinates audio normalization, model selection, STT, and speaker tagging."""

    def __init__(
        self,
        loader: ModelLoader,
        speaker_recognizer: Optional[SpeakerRecognizer] = None,
    ) -> None:
        self._loader = loader
        self._speaker_recognizer = speaker_recognizer

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

    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        """Process a request through the shared service pipeline."""
        request = self._resolve_request(request)
        resolved_model, transcriber = await self._loader.load_transcriber(
            language=request.language, requested_model=request.model
        )

        normalized_audio = None
        if (self.provider != "openai") or request.speaker_enabled:
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
            )


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
