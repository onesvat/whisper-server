"""OpenAI-compatible HTTP API."""

from __future__ import annotations

from dataclasses import asdict
from typing import Annotated

from fastapi import (
    FastAPI,
    File,
    Form,
    UploadFile,
    Depends,
    HTTPException,
    Security,
    Request,
)
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security.api_key import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

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

API_KEY_HEADER = APIKeyHeader(name="Authorization", auto_error=False)


def _get_api_key_identifier(request: Request) -> str:
    """Identify requester by API key (fallback to IP)."""
    auth_header = request.headers.get("Authorization")
    if auth_header:
        if auth_header.lower().startswith("bearer "):
            return auth_header[7:].strip()
        return auth_header.strip()
    return get_remote_address(request)


def create_app(service: SpeechService) -> FastAPI:
    """Build the HTTP API app."""
    limiter = Limiter(key_func=_get_api_key_identifier)
    app = FastAPI(title="whisper-server", version=__version__)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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

    async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
        if not service._key_manager:
            return None

        actual_key = api_key
        if api_key and api_key.lower().startswith("bearer "):
            actual_key = api_key[7:].strip()

        if not actual_key or not service._key_manager.get_key_config(actual_key):
            raise HTTPException(status_code=403, detail="Invalid or missing API Key")
        return actual_key

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        """List all available models and their status."""
        models = service.get_models()
        # Return in a format similar to OpenAI
        return {
            "object": "list",
            "data": [
                {
                    "id": m["id"],
                    "object": "model",
                    "created": 1677610602,  # Dummy timestamp
                    "owned_by": "whisper-server",
                    "ready": m["ready"],
                    "downloaded": m["downloaded"],
                    "available": m["available"],
                    "model_id": m["model_id"],
                }
                for m in models
            ]
        }

    @app.post("/v1/models/{model_id}/unload")
    async def unload_model(model_id: str, api_key: str = Depends(verify_api_key)):
        """Manually unload a model from memory."""
        success = await service.unload_model(model_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not loaded")
        return {"status": "unloaded", "model": model_id}

    def _get_rate_limit(key: str) -> str:
        """Dynamic rate limit based on API key."""
        if not service._key_manager:
            return "1000/minute"

        config = service._key_manager.get_key_config(key)
        if config:
            return config.rate_limit
        return "1/minute"

    @app.post("/v1/audio/transcriptions")
    @limiter.limit(_get_rate_limit)
    async def create_transcription(
        request: Request,
        file: Annotated[UploadFile, File(...)],
        model: Annotated[str, Form(...)],
        language: Annotated[str | None, Form()] = None,
        prompt: Annotated[str | None, Form()] = None,
        response_format: Annotated[str, Form()] = "json",
        vad_filter: Annotated[bool, Form()] = True,
        word_timestamps: Annotated[bool, Form()] = False,
        llm_correct: Annotated[bool, Form()] = False,
        llm_prompt: Annotated[str | None, Form()] = None,
        api_key: str = Depends(verify_api_key),
    ):
        return await _handle_audio_request(
            service=service,
            task=Task.TRANSCRIBE,
            file=file,
            model=model,
            language=language,
            prompt=prompt,
            response_format=response_format,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
            llm_correct=llm_correct,
            llm_prompt=llm_prompt,
            api_key=api_key,
        )

    @app.post("/v1/audio/translations")
    @limiter.limit(_get_rate_limit)
    async def create_translation(
        request: Request,
        file: Annotated[UploadFile, File(...)],
        model: Annotated[str, Form(...)],
        language: Annotated[str | None, Form()] = None,
        prompt: Annotated[str | None, Form()] = None,
        response_format: Annotated[str, Form()] = "json",
        vad_filter: Annotated[bool, Form()] = True,
        word_timestamps: Annotated[bool, Form()] = False,
        llm_correct: Annotated[bool, Form()] = False,
        llm_prompt: Annotated[str | None, Form()] = None,
        api_key: str = Depends(verify_api_key),
    ):
        return await _handle_audio_request(
            service=service,
            task=Task.TRANSLATE,
            file=file,
            model=model,
            language=language,
            prompt=prompt,
            response_format=response_format,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
            llm_correct=llm_correct,
            llm_prompt=llm_prompt,
            api_key=api_key,
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
    vad_filter: bool,
    word_timestamps: bool,
    llm_correct: bool,
    llm_prompt: str | None,
    api_key: str | None = None,
):
    _validate_response_format(response_format)
    _validate_verbose_json_options(
        service=service,
        model=model,
        response_format=response_format,
        word_timestamps=word_timestamps,
        llm_correct=llm_correct,
    )

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
        vad_filter=vad_filter,
        word_timestamps=word_timestamps,
        llm_correct=llm_correct,
        llm_prompt=llm_prompt,
    )
    result = await service.transcribe(request, api_key=api_key)

    if response_format == "text":
        return PlainTextResponse(result.text)

    if response_format == "verbose_json":
        return JSONResponse(_serialize_verbose_json(task, result))

    return JSONResponse({"text": result.text})


def _validate_response_format(response_format: str) -> None:
    if response_format in SUPPORTED_RESPONSE_FORMATS:
        return

    if response_format in UNSUPPORTED_RESPONSE_FORMATS:
        raise InvalidTranscriptionRequest(
            f"response_format '{response_format}' is not supported in v1"
        )

    raise InvalidTranscriptionRequest(f"unknown response_format '{response_format}'")


def _validate_verbose_json_options(
    *,
    service: SpeechService,
    model: str,
    response_format: str,
    word_timestamps: bool,
    llm_correct: bool,
) -> None:
    if word_timestamps and response_format != "verbose_json":
        raise InvalidTranscriptionRequest(
            "word_timestamps=true requires response_format=verbose_json"
        )

    if response_format != "verbose_json":
        return

    if service.provider == "openai":
        raise InvalidTranscriptionRequest(
            "response_format 'verbose_json' is only supported for the local faster-whisper provider"
        )

    if llm_correct:
        raise InvalidTranscriptionRequest(
            "response_format 'verbose_json' cannot be combined with llm_correct=true"
        )

    if model.startswith("speaker:"):
        raise InvalidTranscriptionRequest(
            "response_format 'verbose_json' cannot be combined with speaker: model aliases"
        )


def _serialize_verbose_json(task: Task, result) -> dict[str, object]:
    payload: dict[str, object] = {
        "task": task.value,
        "language": result.language,
        "duration": result.duration,
        "text": result.text,
        "segments": [],
    }

    for segment in result.segments or []:
        segment_payload = asdict(segment)
        if segment.words is None:
            segment_payload.pop("words", None)
        payload["segments"].append(segment_payload)

    return payload
