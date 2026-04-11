"""OpenAI-compatible HTTP API."""

from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from . import __version__
from .const import (
    AudioInput,
    SUPPORTED_RESPONSE_FORMATS,
    Task,
    TranscriptionRequest,
    UNSUPPORTED_RESPONSE_FORMATS,
)
from .models import is_valid_model_name
from .service import InvalidTranscriptionRequest, SpeechService


def create_app(service: SpeechService) -> FastAPI:
    """Build the HTTP API app."""
    app = FastAPI(title="whisper-server", version=__version__)

    @app.exception_handler(InvalidTranscriptionRequest)
    async def invalid_request_handler(_request, exc: InvalidTranscriptionRequest):
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": str(exc),
                    "type": "invalid_request_error",
                }
            },
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/audio/transcriptions")
    async def create_transcription(
        file: Annotated[UploadFile, File(...)],
        model: Annotated[str, Form(...)],
        language: Annotated[str | None, Form()] = None,
        prompt: Annotated[str | None, Form()] = None,
        response_format: Annotated[str, Form()] = "json",
    ):
        return await _handle_audio_request(
            service=service,
            task=Task.TRANSCRIBE,
            file=file,
            model=model,
            language=language,
            prompt=prompt,
            response_format=response_format,
        )

    @app.post("/v1/audio/translations")
    async def create_translation(
        file: Annotated[UploadFile, File(...)],
        model: Annotated[str, Form(...)],
        language: Annotated[str | None, Form()] = None,
        prompt: Annotated[str | None, Form()] = None,
        response_format: Annotated[str, Form()] = "json",
    ):
        return await _handle_audio_request(
            service=service,
            task=Task.TRANSLATE,
            file=file,
            model=model,
            language=language,
            prompt=prompt,
            response_format=response_format,
        )

    return app


async def _handle_audio_request(
    *,
    service: SpeechService,
    task: Task,
    file: UploadFile,
    model: str,
    language: str | None,
    prompt: str | None,
    response_format: str,
):
    _validate_response_format(response_format)

    if not is_valid_model_name(model):
        raise InvalidTranscriptionRequest(
            f"Invalid model '{model}'. Use a predefined alias (tiny, base, medium, large, turbo), "
            f"a HuggingFace model ID (Organization/model-name), or a local path."
        )

    audio_bytes = await file.read()
    if not audio_bytes:
        raise InvalidTranscriptionRequest("uploaded file is empty")

    request = TranscriptionRequest(
        audio=AudioInput(
            data=audio_bytes,
            filename=file.filename or "audio.wav",
            content_type=file.content_type,
        ),
        model=model,
        language=language,
        task=task,
        initial_prompt=prompt,
    )
    result = await service.transcribe(request)

    if response_format == "text":
        return PlainTextResponse(result.text)

    return JSONResponse({"text": result.text})


def _validate_response_format(response_format: str) -> None:
    if response_format in SUPPORTED_RESPONSE_FORMATS:
        return

    if response_format in UNSUPPORTED_RESPONSE_FORMATS:
        raise InvalidTranscriptionRequest(
            f"response_format '{response_format}' is not supported in v1"
        )

    raise InvalidTranscriptionRequest(f"unknown response_format '{response_format}'")
