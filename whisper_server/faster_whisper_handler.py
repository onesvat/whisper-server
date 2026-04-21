"""Local faster-whisper transcriber."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import faster_whisper

from .const import (
    Task,
    Transcriber,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionWord,
)


class FasterWhisperTranscriber(Transcriber):
    """Transcriber backed by faster-whisper."""

    def __init__(
        self,
        model_id: str,
        cache_dir: Union[str, Path],
        device: str = "cpu",
        compute_type: str = "default",
        cpu_threads: int = 4,
        vad_parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.vad_filter = vad_parameters is not None
        self.vad_parameters = vad_parameters
        self.compute_type = compute_type
        self.device = device
        self.model = faster_whisper.WhisperModel(
            model_id,
            download_root=str(cache_dir),
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )

    def transcribe(
        self,
        audio,
        language: Optional[str],
        task: Task = Task.TRANSCRIBE,
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
        vad_filter: Optional[bool] = None,
        word_timestamps: bool = False,
    ) -> TranscriptionResult:
        if vad_filter is None:
            vad_filter = self.vad_filter

        segments, info = self.model.transcribe(
            audio,
            language=language,
            task=task.value,
            beam_size=beam_size,
            initial_prompt=initial_prompt,
            vad_filter=vad_filter,
            vad_parameters=self.vad_parameters,
            word_timestamps=word_timestamps,
        )
        serialized_segments: list[TranscriptionSegment] = []
        text_parts: list[str] = []

        for segment in segments:
            segment_text = segment.text.strip()
            if segment_text:
                text_parts.append(segment_text)

            words = None
            if word_timestamps:
                words = [
                    TranscriptionWord(
                        word=word.word,
                        start=word.start,
                        end=word.end,
                        probability=getattr(word, "probability", None),
                    )
                    for word in (segment.words or [])
                ]

            serialized_segments.append(
                TranscriptionSegment(
                    id=segment.id,
                    seek=segment.seek,
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    tokens=list(segment.tokens),
                    temperature=segment.temperature,
                    avg_logprob=segment.avg_logprob,
                    compression_ratio=segment.compression_ratio,
                    no_speech_prob=segment.no_speech_prob,
                    words=words,
                )
            )

        return TranscriptionResult(
            text=" ".join(text_parts),
            model=getattr(self.model, "model_size_or_path", "faster-whisper"),
            language=getattr(info, "language", None),
            duration=getattr(info, "duration", None),
            segments=serialized_segments,
        )
