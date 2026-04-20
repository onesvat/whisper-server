from fastapi.testclient import TestClient

from whisper_server.const import Task, TranscriptionResult
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
        return TranscriptionResult(text="selam", model="base")


def _multipart(model="base", response_format="json"):
    return {
        "file": ("audio.wav", b"fake wav bytes", "audio/wav"),
        "model": (None, model),
        "response_format": (None, response_format),
    }


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


def test_unsupported_response_format_returns_400():
    service = RecordingService()
    client = TestClient(create_app(service))

    response = client.post(
        "/v1/audio/transcriptions",
        files=_multipart(response_format="verbose_json"),
    )

    assert response.status_code == 400
    assert "not supported" in response.json()["error"]["message"]


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
