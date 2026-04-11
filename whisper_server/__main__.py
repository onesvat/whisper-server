#!/usr/bin/env python3
"""CLI entrypoint for whisper-server."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import platform
import subprocess
from functools import partial
from typing import Any, Dict, Optional

import faster_whisper
import uvicorn
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer, AsyncTcpServer

from . import __version__
from .const import AUTO_LANGUAGE, AUTO_MODEL, SttLibrary
from .http_api import create_app
from .models import ModelLoader, normalize_model_name
from .service import SpeechService, parse_model_alias
from .speaker_recognition import create_speaker_recognizer_from_env
from .wyoming_handler import WyomingEventHandler

_LOGGER = logging.getLogger(__name__)


def _detect_device() -> str:
    """Auto-detect CUDA availability."""
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return "cuda"
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True, timeout=2)
        return "cuda"
    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        return "cpu"


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", help="unix:// or tcp:// for the Wyoming listener")
    parser.add_argument(
        "--openai-http-host",
        help="Host for the OpenAI-compatible HTTP listener",
    )
    parser.add_argument(
        "--openai-http-port",
        type=int,
        help="Port for the OpenAI-compatible HTTP listener",
    )
    parser.add_argument(
        "--zeroconf",
        nargs="?",
        const="whisper-server",
        help="Enable discovery over zeroconf with optional name (default: whisper-server)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("WHISPER_MODEL", AUTO_MODEL),
        help=f"Name of model to use (or {AUTO_MODEL})",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        action="append",
        help="Data directory to check for downloaded models",
    )
    parser.add_argument(
        "--download-dir",
        help="Directory to download models into (default: first data dir)",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("WHISPER_DEVICE"),
        help="Device to use for inference (default: auto-detect cuda/cpu)",
    )
    parser.add_argument(
        "--language",
        default=os.environ.get("WHISPER_LANGUAGE", AUTO_LANGUAGE),
        help=f"Default language to set for transcription (default: {AUTO_LANGUAGE})",
    )
    parser.add_argument(
        "--compute-type",
        default=os.environ.get("WHISPER_COMPUTE_TYPE", "default"),
        help="Compute type (float16, int8, etc.)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=int(os.environ.get("WHISPER_BEAM_SIZE", "0")),
        help="Size of beam during decoding (0 for auto)",
    )
    parser.add_argument(
        "--cpu-threads",
        default=int(os.environ.get("WHISPER_CPU_THREADS", "4")),
        type=int,
        help="Number of CPU threads to use for inference (default: 4)",
    )
    parser.add_argument(
        "--initial-prompt",
        default=os.environ.get("WHISPER_INITIAL_PROMPT"),
        help="Optional text prompt for transcription requests",
    )
    parser.add_argument(
        "--vad-filter",
        action="store_true",
        help="Enable Silero VAD to reduce hallucinations (local provider only)",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="VAD speech probability threshold (default: 0.5)",
    )
    parser.add_argument(
        "--vad-min-speech-ms",
        type=int,
        default=250,
        help="VAD minimum speech duration in ms (default: 250)",
    )
    parser.add_argument(
        "--vad-min-silence-ms",
        type=int,
        default=2000,
        help="VAD minimum silence duration in ms to split (default: 2000)",
    )
    parser.add_argument(
        "--stt-library",
        choices=[lib.value for lib in SttLibrary],
        default=SttLibrary.AUTO,
        help="Set library to use for speech-to-text",
    )
    parser.add_argument(
        "--provider",
        choices=["local", "openai"],
        default=os.environ.get("WHISPER_PROVIDER", "local"),
        help="Transcription provider: local or openai (default: local)",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Don't check HuggingFace hub for model updates",
    )
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    parser.add_argument(
        "--log-format", default=logging.BASIC_FORMAT, help="Format for log messages"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print version and exit",
    )
    args = parser.parse_args()

    if args.device is None:
        args.device = _detect_device()
        _LOGGER.info("Auto-detected device: %s", args.device)

    http_enabled = (args.openai_http_host is not None) or (
        args.openai_http_port is not None
    )
    if (args.uri is None) and (not http_enabled):
        parser.error("configure at least one listener via --uri or --openai-http-port")

    if http_enabled:
        args.openai_http_host = args.openai_http_host or "0.0.0.0"
        args.openai_http_port = args.openai_http_port or 8080

    if not args.download_dir:
        args.download_dir = args.data_dir[0]

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format
    )
    _LOGGER.debug(args)

    args.stt_library = SttLibrary(args.stt_library)
    machine = platform.machine().lower()
    is_arm = ("arm" in machine) or ("aarch" in machine)
    if args.beam_size <= 0:
        args.beam_size = 1 if is_arm else 5
        _LOGGER.debug("Beam size automatically selected: %s", args.beam_size)

    if args.language == AUTO_LANGUAGE:
        args.language = None
    if args.model == AUTO_MODEL:
        args.model = None

    selection = parse_model_alias(args.model)
    display_model = (
        normalize_model_name(selection.model) if selection.model else AUTO_MODEL
    )

    vad_parameters: Optional[Dict[str, Any]] = None
    if args.vad_filter:
        vad_parameters = {
            "threshold": args.vad_threshold,
            "min_speech_duration_ms": args.vad_min_speech_ms,
            "min_silence_duration_ms": args.vad_min_silence_ms,
        }

    loader = ModelLoader(
        preferred_stt_library=args.stt_library,
        preferred_language=args.language,
        download_dir=args.download_dir,
        local_files_only=args.local_files_only,
        model=args.model,
        compute_type=args.compute_type,
        device=args.device,
        beam_size=args.beam_size,
        cpu_threads=args.cpu_threads,
        initial_prompt=args.initial_prompt,
        vad_parameters=vad_parameters,
        provider=args.provider,
    )
    service = SpeechService(
        loader=loader,
        speaker_recognizer=create_speaker_recognizer_from_env(),
    )

    _LOGGER.debug("Pre-loading transcriber")
    await service.warmup(model=args.model, language=args.language)

    wyoming_info = _build_wyoming_info(display_model)

    async with asyncio.TaskGroup() as task_group:
        if args.uri:
            task_group.create_task(
                _run_wyoming_server(
                    uri=args.uri,
                    zeroconf_name=args.zeroconf,
                    wyoming_info=wyoming_info,
                    service=service,
                )
            )
        if http_enabled:
            task_group.create_task(
                _run_http_server(
                    service=service,
                    host=args.openai_http_host,
                    port=args.openai_http_port,
                    debug=args.debug,
                )
            )


def _build_wyoming_info(model_name: str) -> Info:
    return Info(
        asr=[
            AsrProgram(
                name="whisper-server",
                description="Shared Wyoming and OpenAI-compatible speech-to-text service",
                attribution=Attribution(
                    name="Guillaume Klein",
                    url="https://github.com/guillaumekln/faster-whisper/",
                ),
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name=model_name,
                        description=model_name,
                        attribution=Attribution(
                            name="Systran",
                            url="https://huggingface.co/Systran",
                        ),
                        installed=True,
                        languages=sorted(
                            list(set(faster_whisper.tokenizer._LANGUAGE_CODES))
                        ),
                        version=faster_whisper.__version__,
                    )
                ],
            )
        ],
    )


async def _run_wyoming_server(
    *,
    uri: str,
    zeroconf_name: Optional[str],
    wyoming_info: Info,
    service: SpeechService,
) -> None:
    server = AsyncServer.from_uri(uri)

    if zeroconf_name:
        if not isinstance(server, AsyncTcpServer):
            raise ValueError("Zeroconf requires a tcp:// Wyoming uri")

        from wyoming.zeroconf import HomeAssistantZeroconf

        tcp_server: AsyncTcpServer = server
        hass_zeroconf = HomeAssistantZeroconf(
            name=zeroconf_name, port=tcp_server.port, host=tcp_server.host
        )
        await hass_zeroconf.register_server()
        _LOGGER.debug("Zeroconf discovery enabled")

    _LOGGER.info("Starting Wyoming listener on %s", uri)
    await server.run(partial(WyomingEventHandler, wyoming_info, service))


async def _run_http_server(
    *, service: SpeechService, host: str, port: int, debug: bool
) -> None:
    _LOGGER.info("Starting OpenAI-compatible HTTP listener on %s:%s", host, port)
    config = uvicorn.Config(
        create_app(service),
        host=host,
        port=port,
        log_level="debug" if debug else "info",
    )
    server = uvicorn.Server(config)
    await server.serve()


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
