"""Local faster-whisper transcriber."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import faster_whisper

from .const import Task, Transcriber


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
    ) -> str:
        segments, _info = self.model.transcribe(
            audio,
            language=language,
            task=task.value,
            beam_size=beam_size,
            initial_prompt=initial_prompt,
            vad_filter=self.vad_filter,
            vad_parameters=self.vad_parameters,
        )
        return " ".join(segment.text.strip() for segment in segments if segment.text.strip())
