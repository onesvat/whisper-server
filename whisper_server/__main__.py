#!/usr/bin/env python3
"""CLI entrypoint for whisper-server."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import platform
import subprocess
import sys
from functools import partial
from pathlib import Path
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
    parser = argparse.ArgumentParser(prog="whisper-server")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the STT server")
    serve_parser.add_argument(
        "--uri",
        default=os.environ.get("WHISPER_URI"),
        help="unix:// or tcp:// for the Wyoming listener",
    )
    serve_parser.add_argument(
        "--openai-http-host",
        default=os.environ.get("WHISPER_HTTP_HOST"),
        help="Host for the OpenAI-compatible HTTP listener",
    )
    serve_parser.add_argument(
        "--openai-http-port",
        type=int,
        default=int(os.environ.get("WHISPER_HTTP_PORT", "0")) or None,
        help="Port for the OpenAI-compatible HTTP listener",
    )
    serve_parser.add_argument(
        "--zeroconf",
        nargs="?",
        const="whisper-server",
        default=os.environ.get("WHISPER_ZEROCONF"),
        help="Enable discovery over zeroconf with optional name (default: whisper-server)",
    )
    serve_parser.add_argument(
        "--model",
        default=os.environ.get("WHISPER_MODEL", AUTO_MODEL),
        help=f"Name of model to use (or {AUTO_MODEL})",
    )
    serve_parser.add_argument(
        "--data-dir",
        action="append",
        default=os.environ.get("WHISPER_DATA_DIR", "").split(",") if os.environ.get("WHISPER_DATA_DIR") else [],
        help="Data directory to check for downloaded models",
    )
    serve_parser.add_argument(
        "--download-dir",
        default=os.environ.get("WHISPER_DOWNLOAD_DIR"),
        help="Directory to download models into (default: first data dir)",
    )
    serve_parser.add_argument(
        "--device",
        default=os.environ.get("WHISPER_DEVICE"),
        help="Device to use for inference (default: auto-detect cuda/cpu)",
    )
    serve_parser.add_argument(
        "--language",
        default=os.environ.get("WHISPER_LANGUAGE", AUTO_LANGUAGE),
        help=f"Default language to set for transcription (default: {AUTO_LANGUAGE})",
    )
    serve_parser.add_argument(
        "--compute-type",
        default=os.environ.get("WHISPER_COMPUTE_TYPE", "default"),
        help="Compute type (float16, int8, etc.)",
    )
    serve_parser.add_argument(
        "--beam-size",
        type=int,
        default=int(os.environ.get("WHISPER_BEAM_SIZE", "0")),
        help="Size of beam during decoding (0 for auto)",
    )
    serve_parser.add_argument(
        "--cpu-threads",
        default=int(os.environ.get("WHISPER_CPU_THREADS", "4")),
        type=int,
        help="Number of CPU threads to use for inference (default: 4)",
    )
    serve_parser.add_argument(
        "--initial-prompt",
        default=os.environ.get("WHISPER_INITIAL_PROMPT"),
        help="Optional text prompt for transcription requests",
    )
    serve_parser.add_argument(
        "--vad-filter",
        action="store_true",
        default=os.environ.get("WHISPER_VAD_FILTER", "").lower() == "true",
        help="Enable Silero VAD to reduce hallucinations (local provider only)",
    )
    serve_parser.add_argument(
        "--vad-threshold",
        type=float,
        default=float(os.environ.get("WHISPER_VAD_THRESHOLD", "0.5")),
        help="VAD speech probability threshold (default: 0.5)",
    )
    serve_parser.add_argument(
        "--vad-min-speech-ms",
        type=int,
        default=int(os.environ.get("WHISPER_VAD_MIN_SPEECH_MS", "250")),
        help="VAD minimum speech duration in ms (default: 250)",
    )
    serve_parser.add_argument(
        "--vad-min-silence-ms",
        type=int,
        default=int(os.environ.get("WHISPER_VAD_MIN_SILENCE_MS", "2000")),
        help="VAD minimum silence duration in ms to split (default: 2000)",
    )
    serve_parser.add_argument(
        "--stt-library",
        choices=[lib.value for lib in SttLibrary],
        default=os.environ.get("WHISPER_STT_LIBRARY", SttLibrary.AUTO.value),
        help="Set library to use for speech-to-text",
    )
    serve_parser.add_argument(
        "--provider",
        choices=["local", "openai"],
        default=os.environ.get("WHISPER_PROVIDER", "local"),
        help="Transcription provider: local or openai (default: local)",
    )
    serve_parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=os.environ.get("WHISPER_LOCAL_FILES_ONLY", "").lower() == "true",
        help="Don't check HuggingFace hub for model updates",
    )
    serve_parser.add_argument(
        "--keys-file",
        default=os.environ.get("WHISPER_KEYS_FILE"),
        help="JSON file containing API keys and their rate limits",
    )
    serve_parser.add_argument(
        "--stats-db",
        default=os.environ.get("WHISPER_STATS_DB"),
        help="SQLite file for usage statistics",
    )
    serve_parser.add_argument(
        "--storage-dir",
        default=os.environ.get("WHISPER_STORAGE_DIR"),
        help="Directory to save audio and transcript logs",
    )
    serve_parser.add_argument(
        "--retention-days",
        type=int,
        default=int(os.environ.get("WHISPER_RETENTION_DAYS", "30")),
        help="Number of days to keep stored audio (default: 30)",
    )
    serve_parser.add_argument(
        "--retention-max-gb",
        type=float,
        default=float(os.environ.get("WHISPER_RETENTION_MAX_GB", "10.0")),
        help="Maximum size of storage directory in GB (default: 10.0)",
    )
    serve_parser.add_argument(
        "--model-ttl",
        type=int,
        default=int(os.environ.get("WHISPER_MODEL_TTL", "30")),
        help="Number of minutes to keep an idle model in memory (default: 30, 0 to disable)",
    )
    serve_parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    serve_parser.add_argument(
        "--log-format", default=logging.BASIC_FORMAT, help="Format for log messages"
    )

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show usage statistics")
    stats_parser.add_argument(
        "--stats-db",
        default=os.environ.get("WHISPER_STATS_DB"),
        help="Path to the statistics SQLite database",
    )
    stats_parser.add_argument(
        "--keys-file",
        default=os.environ.get("WHISPER_KEYS_FILE"),
        help="Optional JSON file containing API keys to map names",
    )
    stats_parser.add_argument(
        "--days",
        type=int,
        default=int(os.environ.get("WHISPER_STATS_DAYS", "7")),
        help="Show stats for the last N days (default: 7)",
    )

    # Keys command
    keys_parser = subparsers.add_parser("keys", help="Manage API keys")
    keys_parser.add_argument(
        "--keys-file",
        default=os.environ.get("WHISPER_KEYS_FILE"),
        help="Path to the JSON keys file",
    )
    keys_subparsers = keys_parser.add_subparsers(dest="action", help="Action to perform")
    
    keys_subparsers.add_parser("list", help="List all keys")
    
    add_key_parser = keys_subparsers.add_parser("add", help="Add or update a key")
    add_key_parser.add_argument("--key", required=True, help="The API key string")
    add_key_parser.add_argument("--name", required=True, help="Friendly name for the key owner")
    add_key_parser.add_argument("--limit", default="30/minute", help="Rate limit")
    
    del_key_parser = keys_subparsers.add_parser("delete", help="Delete a key")
    del_key_parser.add_argument("--key", required=True, help="The API key string to remove")

    # If no command is provided via CLI, check environment variable
    if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] not in ["serve", "stats", "keys"]):
        env_cmd = os.environ.get("WHISPER_CMD")
        if env_cmd in ["serve", "stats", "keys"]:
            # Insert the command into sys.argv so argparse sees it
            sys.argv.insert(1, env_cmd)

    args = parser.parse_args()

    if args.command == "keys":
        if not args.keys_file:
            print("Error: --keys-file or WHISPER_KEYS_FILE is required")
            return
        from .key_manager import KeyManager
        key_manager = KeyManager(Path(args.keys_file), Path("none"))
        if args.action == "list":
            key_manager.list_keys()
        elif args.action == "add":
            key_manager.add_key(args.key, args.name, args.limit)
            print(f"Key added/updated: {args.name}")
        elif args.action == "delete":
            if key_manager.delete_key(args.key):
                print(f"Key deleted successfully.")
            else:
                print("Key not found.")
        else:
            keys_parser.print_help()
        return

    if args.command == "stats":
        if not args.stats_db:
            print("Error: --stats-db or WHISPER_STATS_DB is required")
            return
        from .key_manager import KeyManager
        key_manager = KeyManager(Path(args.keys_file) if args.keys_file else Path("none"), Path(args.stats_db))
        await key_manager.print_stats_table(days=args.days)
        return

    if args.command != "serve":
        parser.print_help()
        return

    if args.device is None:
        args.device = _detect_device()
        _LOGGER.info("Auto-detected device: %s", args.device)

    http_enabled = (args.openai_http_host is not None) or (
        args.openai_http_port is not None
    )
    if (args.uri is None) and (not http_enabled):
        # Fallback to defaults if neither is specified
        args.openai_http_host = args.openai_http_host or "0.0.0.0"
        args.openai_http_port = args.openai_http_port or 8080
        http_enabled = True

    if not args.download_dir and args.data_dir:
        args.download_dir = args.data_dir[0]
    elif not args.download_dir:
        # Emergency default
        args.download_dir = "/data"

    if not args.data_dir:
        args.data_dir = [args.download_dir]

    key_manager = None
    if args.keys_file:
        keys_path = Path(args.keys_file)
        stats_db_path = Path(args.stats_db) if args.stats_db else Path(args.download_dir) / "stats.db"
        from .key_manager import KeyManager
        key_manager = KeyManager(keys_path, stats_db_path)
        await key_manager.setup_db()

    storage_manager = None
    if args.storage_dir:
        from .storage import StorageManager
        storage_manager = StorageManager(
            base_dir=Path(args.storage_dir),
            retention_days=args.retention_days,
            max_gb=args.retention_max_gb,
        )

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
        key_manager=key_manager,
        storage_manager=storage_manager,
    )

    _LOGGER.debug("Pre-loading transcriber")
    await service.warmup(model=args.model, language=args.language)

    wyoming_info = _build_wyoming_info(display_model)

async with asyncio.TaskGroup() as task_group:
        if storage_manager:
            task_group.create_task(storage_manager.run_retention_loop())
        
        if args.model_ttl > 0:
            _LOGGER.info("Idle model unloading enabled (TTL: %d minutes)", args.model_ttl)
            task_group.create_task(_run_model_unload_loop(service, args.model_ttl * 60))

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


async def _run_model_unload_loop(service: SpeechService, ttl_seconds: float) -> None:
    """Periodically check for and unload idle models."""
    while True:
        try:
            await asyncio.sleep(60)  # Check every minute
            await service.unload_idle_models(ttl_seconds)
        except Exception as err:
            _LOGGER.error("Error in model unload loop: %s", err)


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
