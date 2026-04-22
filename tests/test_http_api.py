from fastapi.testclient import TestClient

from whisper_server.const import Task, TranscriptionResult, TranscriptionSegment, TranscriptionWord
from whisper_server.http_api import create_app
from whisper_server.service import InvalidTranscriptionRequest


class RecordingService:
    def __init__(self) -> None:
        self.calls = []
        self._key_manager = None

    async def transcribe(self, request, api_key=None):
        if request.model == "speaker:":
            raise InvalidTranscriptionRequest(
                "speaker: model alias requires a model name"
            )

        self.calls.append(request)
        segments = [
            TranscriptionSegment(
                id=0,
                seek=0,
                start=0.0,
                end=1.2,
                text=" selam",
                tokens=[1, 2],
                temperature=0.0,
                avg_logprob=-0.1,
                compression_ratio=1.1,
                no_speech_prob=0.01,
                words=(
                    [
                        TranscriptionWord(
                            word="selam",
                            start=0.0,
                            end=0.5,
                            probability=0.95,
                        )
                    ]
                    if request.word_timestamps
                    else None
                ),
            )
        ]
        return TranscriptionResult(
            text="selam",
            model="base",
            language="tr",
            duration=1.2,
            segments=segments,
        )


def _multipart(model="base", response_format="json", **extra_fields):
    data = {
        "file": ("audio.wav", b"fake wav bytes", "audio/wav"),
        "model": (None, model),
        "response_format": (None, response_format),
    }
    for key, value in extra_fields.items():
        if isinstance(value, bool):
            value = "true" if value else "false"
        data[key] = (None, str(value))
    return data


def test_transcriptions_json_response():
    service = RecordingService()
    client = TestClient(create_app(service))

    response = client.post("/v1/audio/transcriptions", files=_multipart())

    assert response.status_code == 200
    assert response.json() == {"text": "selam"}
    assert service.calls[0].task == Task.TRANSCRIBE


def test_transcriptions_text_response():
    service = RecordingService()
    client = TestClient(create_app(service))

    response = client.post(
        "/v1/audio/transcriptions",
        files=_multipart(response_format="text"),
    )

    assert response.status_code == 200
    assert response.text == "selam"


def test_translations_route():
    service = RecordingService()
    client = TestClient(create_app(service))

    response = client.post("/v1/audio/translations", files=_multipart())

    assert response.status_code == 200
    assert service.calls[0].task == Task.TRANSLATE


def test_invalid_speaker_alias_returns_400():
    service = RecordingService()
    client = TestClient(create_app(service))

    response = client.post(
        "/v1/audio/transcriptions",
        files=_multipart(model="speaker:"),
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_verbose_json_response():
    service = RecordingService()
    client = TestClient(create_app(service))

    response = client.post(
        "/v1/audio/transcriptions",
        files=_multipart(response_format="verbose_json"),
    )

    assert response.status_code == 200
    assert response.json() == {
        "task": "transcribe",
        "language": "tr",
        "duration": 1.2,
        "text": "selam",
        "segments": [
            {
                "id": 0,
                "seek": 0,
                "start": 0.0,
                "end": 1.2,
                "text": " selam",
                "tokens": [1, 2],
                "temperature": 0.0,
                "avg_logprob": -0.1,
                "compression_ratio": 1.1,
                "no_speech_prob": 0.01,
            }
        ],
    }


def test_verbose_json_with_word_timestamps():
    service = RecordingService()
    client = TestClient(create_app(service))

    response = client.post(
        "/v1/audio/transcriptions",
        files=_multipart(response_format="verbose_json", word_timestamps=True),
    )

    assert response.status_code == 200
    assert response.json()["segments"][0]["words"] == [
        {
            "word": "selam",
            "start": 0.0,
            "end": 0.5,
            "probability": 0.95,
        }
    ]


def test_word_timestamps_requires_verbose_json():
    service = RecordingService()
    client = TestClient(create_app(service))

    response = client.post(
        "/v1/audio/transcriptions",
        files=_multipart(word_timestamps=True),
    )

    assert response.status_code == 400
    assert "requires response_format=verbose_json" in response.json()["error"]["message"]


def test_invalid_model_name_returns_400():
    service = RecordingService()
    client = TestClient(create_app(service))

    response = client.post(
        "/v1/audio/transcriptions",
        files=_multipart(model="invalid-model-name"),
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "Invalid model" in response.json()["error"]["message"]


def test_custom_huggingface_model_is_allowed():
    """Custom HuggingFace models (with slash) should be accepted."""
    service = RecordingService()
    client = TestClient(create_app(service))

    response = client.post(
        "/v1/audio/transcriptions",
        files=_multipart(model="my-org/my-custom-model"),
    )

    assert response.status_code == 200


def test_new_parameters_passed():
    service = RecordingService()
    client = TestClient(create_app(service))

    response = client.post(
        "/v1/audio/transcriptions",
        files={
            "file": ("audio.wav", b"fake wav bytes", "audio/wav"),
            "model": (None, "base"),
            "vad_filter": (None, "false"),
            "llm_correct": (None, "true"),
            "llm_prompt": (None, "Custom Prompt"),
        },
    )

    assert response.status_code == 200
    assert service.calls[0].vad_filter is False
    assert service.calls[0].llm_correct is True
    assert service.calls[0].llm_prompt == "Custom Prompt"


def test_verbose_json_rejects_llm_correct():
    service = RecordingService()
    client = TestClient(create_app(service))

    response = client.post(
        "/v1/audio/transcriptions",
        files=_multipart(response_format="verbose_json", llm_correct=True),
    )

    assert response.status_code == 400
    assert "cannot be combined with llm_correct=true" in response.json()["error"]["message"]


def test_verbose_json_rejects_speaker_alias():
    service = RecordingService()
    client = TestClient(create_app(service))

    response = client.post(
        "/v1/audio/transcriptions",
        files=_multipart(model="speaker:base", response_format="verbose_json"),
    )

    assert response.status_code == 400
    assert "cannot be combined with speaker:" in response.json()["error"]["message"]


def test_translation_verbose_json_sets_task():
    service = RecordingService()
    client = TestClient(create_app(service))

    response = client.post(
        "/v1/audio/translations",
        files=_multipart(response_format="verbose_json"),
    )

    assert response.status_code == 200
    assert response.json()["task"] == "translate"


def test_plain_json_response_still_allowed():
    service = RecordingService()
    client = TestClient(create_app(service))

    response = client.post("/v1/audio/transcriptions", files=_multipart())

    assert response.status_code == 200
    assert response.json() == {"text": "selam"}
