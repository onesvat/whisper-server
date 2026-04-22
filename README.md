# whisper-server

> Shared Wyoming TCP and OpenAI-compatible HTTP speech-to-text service powered by local `faster-whisper` models.

[![Docker Hub](https://img.shields.io/docker/pulls/onesvat/whisper-server.svg)](https://hub.docker.com/r/onesvat/whisper-server)

## Overview

`whisper-server` exposes two interfaces over the same local STT runtime:

- Wyoming TCP ASR, typically on `10300`
- OpenAI-compatible HTTP audio routes, typically on `8080`

The server does not proxy outbound transcription requests to OpenAI. It runs local `faster-whisper` models directly and keeps the OpenAI-compatible HTTP routes as the client-facing interface.

## Current Capabilities

- Single STT backend: `faster-whisper`
- Local model aliases such as `tiny`, `base`, `small`, `medium`, `large-v3`, and `turbo`
- OpenAI-compatible routes:
  - `POST /v1/audio/transcriptions`
  - `POST /v1/audio/translations`
  - `GET /v1/models`
  - `POST /v1/models/{model_id}/unload`
  - `GET /healthz`
- Shared model cache across Wyoming and HTTP
- Optional Silero VAD
- Optional LLM-based correction with `llm_correct`
- Optional single-speaker matching via `speaker:<model>` aliases
- Optional API keys, per-key rate limits, usage stats, and request storage

## Quick Start

```yaml
# docker-compose.yml
services:
  whisper:
    image: onesvat/whisper-server:gpu
    environment:
      - TZ=Europe/Istanbul
      - LLM_BASE_URL=http://host.docker.internal:1234/v1
    volumes:
      - ./data:/data
    ports:
      - "10300:10300"
      - "8080:8080"
    command:
      [
        "serve",
        "--model",
        "large-v3",
        "--data-dir",
        "/data",
        "--device",
        "cuda",
        "--compute-type",
        "float16",
        "--keys-file",
        "/data/keys.json",
        "--storage-dir",
        "/data/storage",
      ]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

## HTTP API

### Endpoints

- `POST /v1/audio/transcriptions`
- `POST /v1/audio/translations`
- `GET /v1/models`
- `POST /v1/models/{model_id}/unload`
- `GET /healthz`

### Request Shape

The audio routes accept `multipart/form-data`.

Required fields:

- `file`: audio upload
- `model`: local model alias, HuggingFace model ID, local path, or `speaker:<model>`

Optional fields:

| Field | Type | Default | Notes |
|---|---|---|---|
| `language` | string | auto | Optional language hint |
| `prompt` | string | none | Initial prompt for transcription |
| `response_format` | string | `json` | `json`, `text`, `verbose_json` |
| `vad_filter` | bool | `true` | Enable or disable Silero VAD |
| `word_timestamps` | bool | `false` | Only valid with `response_format=verbose_json` |
| `llm_correct` | bool | `false` | Post-process transcript with an LLM |
| `llm_prompt` | string | none | Override the default LLM correction prompt |

### Response Formats

Currently supported:

- `json`
- `text`
- `verbose_json`

Currently not supported:

- `srt`
- `vtt`
- `diarized_json`

Validation rules:

- `word_timestamps=true` requires `response_format=verbose_json`
- `response_format=verbose_json` cannot be combined with `llm_correct=true`
- `response_format=verbose_json` cannot be combined with `speaker:<model>`

### Examples

Basic transcription:

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@sample.wav" \
  -F "model=large-v3"
```

Authenticated request with LLM correction:

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer my-secret-123" \
  -F "file=@sample.wav" \
  -F "model=large-v3" \
  -F "llm_correct=true"
```

Verbose JSON with word timestamps:

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@sample.wav" \
  -F "model=large-v3" \
  -F "response_format=verbose_json" \
  -F "word_timestamps=true"
```

### OpenAI Compatibility Notes

The HTTP routes intentionally resemble OpenAI Audio, but the implementation is local-first and feature-compatible only where it makes sense.

- We keep the OpenAI-style route structure.
- We do not use OpenAI model names.
- We expose custom local features such as `vad_filter`, `llm_correct`, and `speaker:<model>`.
- We do not currently implement streaming transcription, job polling, or diarized responses.

## Wyoming API

The Wyoming listener shares the same `SpeechService` and model cache as HTTP. Audio is collected as PCM, wrapped into WAV, transcribed locally, and returned as a Wyoming transcript event.

## Supported Local Models

VRAM requirements are approximate for `float16` vs `int8`.

| Alias | Full Model ID | VRAM (fp16) | VRAM (int8) | Use Case |
|---|---|---|---|---|
| `tiny` | `Systran/faster-whisper-tiny` | 0.6 GB | 0.4 GB | Testing |
| `base` | `Systran/faster-whisper-base` | 0.7 GB | 0.5 GB | Real-time |
| `small` | `Systran/faster-whisper-small` | 1.1 GB | 0.8 GB | General use |
| `medium` | `Systran/faster-whisper-medium` | 2.5 GB | 1.6 GB | Good accuracy |
| `large-v3` | `Systran/faster-whisper-large-v3` | 4.5 GB | 3.0 GB | Best accuracy |
| `turbo` | `deepdml/faster-whisper-large-v3-turbo-ct2` | 2.6 GB | 1.6 GB | Fast + accurate |
| `distil-large-v3` | `Systran/faster-distil-whisper-large-v3` | 2.4 GB | 1.5 GB | Extreme speed |

## Operations

### API Key Management

```bash
whisper-server keys --keys-file keys.json list
whisper-server keys --keys-file keys.json add --key "my-secret-123" --name "Mobile App" --limit "30/minute"
whisper-server keys --keys-file keys.json delete --key "old-key"
```

### Usage Statistics

```bash
whisper-server stats --stats-db data/stats.db --keys-file data/keys.json
```

### Request Storage

If `--storage-dir` is enabled, the server stores:

- the uploaded audio
- the raw transcription
- the LLM-corrected text when `llm_correct=true`

Stored requests are pruned by retention days and max storage size.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:1234/v1` | Base URL for the LLM correction backend |
| `LLM_API_KEY` | `lm-studio` | API key for the LLM correction backend |
| `LLM_MODEL` | `local-model` | Model name for LLM correction |
| `VOICES_DIR` | `/data/voices` | Directory containing reference speaker clips |
| `SPEAKER_THRESHOLD` | `0.80` | Speaker matching threshold |
| `RETENTION_DAYS` | `30` | Days to keep stored requests |
| `RETENTION_MAX_GB` | `10.0` | Max request storage size in GB |

## CLI

Main subcommands:

- `serve`
- `stats`
- `keys`

Useful `serve` flags:

- `--uri`
- `--openai-http-host`
- `--openai-http-port`
- `--model`
- `--language`
- `--device`
- `--compute-type`
- `--beam-size`
- `--cpu-threads`
- `--vad-filter`
- `--keys-file`
- `--stats-db`
- `--storage-dir`

## Roadmap

Near-term improvements:

- Add `srt` and `vtt` response formats
- Add OpenAI-style `timestamp_granularities` aliases on top of the existing timestamp support
- Improve long-audio handling with chunking and merge logic
- Tighten README and API documentation as the HTTP surface evolves

Potential future work:

- `stream=true` for completed audio uploads
- Optional async job flow for long-running transcriptions
- Better compatibility aliases where they improve client ergonomics
- Real diarization only if a dedicated diarization pipeline is introduced

Explicitly out of scope for now:

- Fake OpenAI model names
- Outbound OpenAI transcription providers
- Pretending current single-speaker matching is full diarization

## License

MIT License
