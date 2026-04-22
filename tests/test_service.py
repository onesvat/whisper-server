import io
import wave
from dataclasses import dataclass

import numpy as np
import pytest

from whisper_server.const import (
    AudioInput,
    Task,
    TranscriptionRequest,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionWord,
)
from whisper_server.service import InvalidTranscriptionRequest, SpeechService, parse_model_alias


def _wav_bytes() -> bytes:
    samples = (np.sin(np.linspace(0, 10, 1600)) * 0.25 * 32767).astype("<i2")
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(samples.tobytes())
        return buffer.getvalue()


@dataclass
class FakeStructuredOutput:
    text: str
    language: str | None = None
    duration: float | None = None
    segments: list[TranscriptionSegment] | None = None


class FakeTranscriber:
    def __init__(self, text: str = "merhaba", structured: bool = False) -> None:
        self.text = text
        self.structured = structured
        self.calls = []

    def transcribe(
        self,
        audio,
        language,
        task=Task.TRANSCRIBE,
        beam_size=5,
        initial_prompt=None,
        vad_filter=None,
        word_timestamps=False,
    ) -> str | TranscriptionResult:
        self.calls.append(
            {
                "audio": audio,
                "language": language,
                "task": task,
                "beam_size": beam_size,
                "initial_prompt": initial_prompt,
                "vad_filter": vad_filter,
                "word_timestamps": word_timestamps,
            }
        )
        if self.structured:
            return TranscriptionResult(
                text=self.text,
                model="ignored-by-service",
                language=language,
                duration=1.5,
                segments=[
                    TranscriptionSegment(
                        id=0,
                        seek=0,
                        start=0.0,
                        end=1.5,
                        text=" merhaba",
                        tokens=[1, 2],
                        temperature=0.0,
                        avg_logprob=-0.2,
                        compression_ratio=1.0,
                        no_speech_prob=0.01,
                        words=(
                            [
                                TranscriptionWord(
                                    word="merhaba",
                                    start=0.0,
                                    end=0.8,
                                    probability=0.9,
                                )
                            ]
                            if word_timestamps
                            else None
                        ),
                    )
                ],
            )
        return self.text


class FakeLoader:
    def __init__(self) -> None:
        self.preferred_language = "tr"
        self.initial_prompt = "startup prompt"
        self.beam_size = 5
        self._transcribers = {}

    async def load_transcriber(self, language=None, requested_model=None):
        model = requested_model or "default-model"
        return model, self._transcribers.setdefault(model, FakeTranscriber())


class FakeSpeakerRecognizer:
    def identify(self, samples):
        return ("onur", 0.95)


def test_parse_model_alias():
    selection = parse_model_alias("speaker:large-v3")
    assert selection.model == "large-v3"
    assert selection.speaker_enabled is True


def test_parse_model_alias_rejects_empty_model():
    with pytest.raises(InvalidTranscriptionRequest):
        parse_model_alias("speaker:")


@pytest.mark.asyncio
async def test_translation_task_routing():
    loader = FakeLoader()
    service = SpeechService(loader=loader)

    result = await service.transcribe(
        TranscriptionRequest(
            audio=AudioInput(data=_wav_bytes(), filename="audio.wav"),
            model="base",
            task=Task.TRANSLATE,
        )
    )

    transcriber = loader._transcribers["base"]
    assert result.text == "merhaba"
    assert transcriber.calls[0]["task"] == Task.TRANSLATE


@pytest.mark.asyncio
async def test_speaker_prefix_only_when_enabled():
    wav_data = _wav_bytes()
    loader = FakeLoader()
    service = SpeechService(loader=loader, speaker_recognizer=FakeSpeakerRecognizer())

    plain_result = await service.transcribe(
        TranscriptionRequest(
            audio=AudioInput(data=wav_data, filename="audio.wav"),
            model="base",
        )
    )
    speaker_result = await service.transcribe(
        TranscriptionRequest(
            audio=AudioInput(data=wav_data, filename="audio.wav"),
            model="speaker:base",
        )
    )

    assert plain_result.text == "merhaba"
    assert plain_result.speaker is None
    assert speaker_result.text == "[onur] merhaba"
    assert speaker_result.speaker == "onur"


@pytest.mark.asyncio
async def test_structured_transcriber_result_is_preserved():
    wav_data = _wav_bytes()
    loader = FakeLoader()
    loader._transcribers["base"] = FakeTranscriber(structured=True)
    service = SpeechService(loader=loader)

    result = await service.transcribe(
        TranscriptionRequest(
            audio=AudioInput(data=wav_data, filename="audio.wav"),
            model="base",
            word_timestamps=True,
        )
    )

    assert result.text == "merhaba"
    assert result.language == "tr"
    assert result.duration == 1.5
    assert result.segments is not None
    assert result.segments[0].words is not None
    assert result.segments[0].words[0].word == "merhaba"


@pytest.mark.asyncio
async def test_plain_transcriber_output_keeps_text_only_behavior():
    wav_data = _wav_bytes()
    loader = FakeLoader()
    service = SpeechService(loader=loader)

    result = await service.transcribe(
        TranscriptionRequest(
            audio=AudioInput(data=wav_data, filename="audio.wav"),
            model="base",
        )
    )

    assert result.text == "merhaba"
    assert result.language is None
    assert result.duration is None
    assert result.segments is None


@pytest.mark.asyncio
async def test_word_timestamps_disabled_leaves_words_empty():
    wav_data = _wav_bytes()
    loader = FakeLoader()
    loader._transcribers["base"] = FakeTranscriber(structured=True)
    service = SpeechService(loader=loader)

    result = await service.transcribe(
        TranscriptionRequest(
            audio=AudioInput(data=wav_data, filename="audio.wav"),
            model="base",
            word_timestamps=False,
        )
    )

    assert result.segments is not None
    assert result.segments[0].words is None
