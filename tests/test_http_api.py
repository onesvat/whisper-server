from fastapi.testclient import TestClient

from whisper_server.const import Task, TranscriptionResult
from whisper_server.http_api import create_app
from whisper_server.service import InvalidTranscriptionRequest


class RecordingService:
    def __init__(self) -> None:
        self.calls = []

    async def transcribe(self, request):
        if request.model == "speaker:":
            raise InvalidTranscriptionRequest("speaker: model alias requires a model name")

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
