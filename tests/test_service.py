import io
import wave

import numpy as np
import pytest

from whisper_server.const import AudioInput, Task, TranscriptionRequest
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


class FakeTranscriber:
    def __init__(self, text: str = "merhaba") -> None:
        self.text = text
        self.calls = []

    def transcribe(
        self,
        audio,
        language,
        task=Task.TRANSCRIBE,
        beam_size=5,
        initial_prompt=None,
    ) -> str:
        self.calls.append(
            {
                "audio": audio,
                "language": language,
                "task": task,
                "beam_size": beam_size,
                "initial_prompt": initial_prompt,
            }
        )
        return self.text


class FakeLoader:
    def __init__(self, provider: str = "local") -> None:
        self.provider = provider
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
