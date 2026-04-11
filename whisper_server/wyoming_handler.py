"""Wyoming protocol adapter."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from .audio import wav_bytes_from_pcm16
from .const import AudioInput, TranscriptionRequest
from .service import SpeechService

_LOGGER = logging.getLogger(__name__)


class WyomingEventHandler(AsyncEventHandler):
    """Dispatches Wyoming ASR requests via the shared service."""

    def __init__(
        self,
        wyoming_info: Info,
        service: SpeechService,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._wyoming_info_event = wyoming_info.event()
        self._service = service
        self._audio_converter = AudioChunkConverter(rate=16000, width=2, channels=1)
        self._pcm_audio = bytearray()
        self._language: Optional[str] = None
        self._warmup_task: Optional[asyncio.Task[str]] = None
        self._save_audio_dir = self._get_save_audio_dir()

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            chunk = self._audio_converter.convert(AudioChunk.from_event(event))
            self._pcm_audio.extend(chunk.audio)

            if self._warmup_task is None:
                self._warmup_task = asyncio.create_task(
                    self._service.warmup(language=self._language)
                )

            return True

        if AudioStop.is_type(event.type):
            return await self._handle_audio_stop()

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            self._language = transcribe.language
            _LOGGER.debug("Language set to %s", self._language)
            return True

        if Describe.is_type(event.type):
            await self.write_event(self._wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        return True

    async def _handle_audio_stop(self) -> bool:
        try:
            if self._warmup_task is not None:
                await self._warmup_task

            wav_bytes = wav_bytes_from_pcm16(bytes(self._pcm_audio))
            result = await self._service.transcribe(
                TranscriptionRequest(
                    audio=AudioInput(
                        data=wav_bytes,
                        filename="speech.wav",
                        content_type="audio/wav",
                    ),
                    language=self._language,
                )
            )
            _LOGGER.info(result.text)
            await self.write_event(Transcript(text=result.text).event())

            if self._save_audio_dir:
                self._save_audio(wav_bytes, result)

            _LOGGER.debug("Completed Wyoming request")
            return False
        finally:
            self._pcm_audio.clear()
            self._language = None
            self._warmup_task = None

    def _get_save_audio_dir(self) -> Optional[Path]:
        save_dir = os.environ.get("SAVE_AUDIO_DIR", "")
        if not save_dir:
            return None

        save_path = Path(save_dir)
        try:
            save_path.mkdir(parents=True, exist_ok=True)
            _LOGGER.info("Saving audio to: %s", save_path)
            return save_path
        except Exception as err:
            _LOGGER.warning("Cannot create save dir %s: %s", save_path, err)
            return None

    def _save_audio(self, wav_bytes: bytes, result) -> None:
        if not self._save_audio_dir:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if result.speaker and result.speaker_score is not None:
            filename = f"{timestamp}_{result.speaker}_{result.speaker_score:.3f}.wav"
        else:
            filename = f"{timestamp}_unknown.wav"

        dest_path = self._save_audio_dir / filename
        try:
            dest_path.write_bytes(wav_bytes)
            dest_path.with_suffix(".txt").write_text(result.text, encoding="utf-8")
            _LOGGER.debug("Saved audio: %s", dest_path.name)
        except Exception as err:
            _LOGGER.warning("Failed to save audio: %s", err)
