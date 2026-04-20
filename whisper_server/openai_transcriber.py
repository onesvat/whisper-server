"""Outbound OpenAI transcription handler."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

from .const import Task, Transcriber

_LOGGER = logging.getLogger(__name__)


class OpenAITranscriber(Transcriber):
    """Transcriber using the OpenAI Audio API."""

    def __init__(self, model_id: str = "gpt-4o-transcribe") -> None:
        self.model_id = model_id
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for the OpenAI provider."
            )

        try:
            from openai import OpenAI
        except ImportError as err:
            raise ImportError(
                "openai package not installed. Install project dependencies first."
            ) from err

        self._client = OpenAI(api_key=self.api_key)
        _LOGGER.info("Initialized OpenAI transcriber with model: %s", model_id)

    def transcribe(
        self,
        audio: Union[str, Path],
        language: Optional[str],
        task: Task = Task.TRANSCRIBE,
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
        vad_filter: Optional[bool] = None,
    ) -> str:
        del beam_size
        del vad_filter

        audio_path = Path(audio)
        with audio_path.open("rb") as audio_file:
            kwargs = {"model": self.model_id, "file": audio_file}
            if initial_prompt:
                kwargs["prompt"] = initial_prompt

            if task == Task.TRANSLATE:
                response = self._client.audio.translations.create(**kwargs)
            else:
                if language:
                    kwargs["language"] = language
                response = self._client.audio.transcriptions.create(**kwargs)

        return response.text
