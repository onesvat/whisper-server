# whisper-server

> Shared Wyoming TCP and OpenAI-compatible HTTP speech-to-text service with optional speaker tagging, VAD, and LLM correction.

[![Docker Hub](https://img.shields.io/docker/pulls/onesvat/whisper-server.svg)](https://hub.docker.com/r/onesvat/whisper-server)

## ✨ Features

- **Two Protocols, One Process**
  - Wyoming TCP ASR on `10300`
  - OpenAI-compatible HTTP STT on `8080`
  - Shared model cache across both listeners

- **Two Transcription Providers**
  - 🏠 **Local**: Faster-Whisper with CTranslate2
  - ☁️ **OpenAI**: outbound OpenAI transcription and translation APIs

- **🚀 Advanced Processing**
  - **Silero VAD**: Built-in Voice Activity Detection to reduce hallucinations and skip silence.
  - **LLM Correction**: Refine transcripts using local (LM Studio) or OpenAI-compatible LLMs.
  - **Speaker Recognition**: Identify speakers based on reference voice samples.

- **🔒 Production Ready**
  - **Authentication**: Secure your API with multiple API keys (JSON-based).
  - **Rate Limiting**: Per-key request limits to prevent abuse.
  - **Usage Stats**: Track requests per key/hour in a local SQLite database.
  - **Data Retention**: Automatically save audio/transcripts with auto-cleanup (FIFO).

## 🚀 Quick Start

### Option 1: Local Transcription (GPU Required)

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
    command: [
      "serve",
      "--model", "large-v3",
      "--data-dir", "/data",
      "--device", "cuda",
      "--compute-type", "float16",
      "--keys-file", "/data/keys.json",
      "--storage-dir", "/data/storage"
    ]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

## 📖 New Features Guide

### 1. VAD (Voice Activity Detection)
The server uses Silero VAD to filter out non-speech segments. This is enabled by default for local providers.
*   **API Toggle:** Pass `vad_filter=false` in the form-data to disable it for a specific request.

### 2. LLM Post-Processing
Refine your transcripts to fix grammar and punctuation using an external LLM.
*   **Setup:** Set `LLM_BASE_URL` and `LLM_MODEL` environment variables.
*   **Usage:** Pass `llm_correct=true` in your API request.
*   **Custom Prompts:** Pass `llm_prompt="Your custom instructions"` to override the default correction style.

### 3. API Key Management
Issue and manage multiple keys via the CLI:
```bash
# List all keys
whisper-server keys --keys-file keys.json list

# Add a key for a mobile app
whisper-server keys --keys-file keys.json add --key "my-secret-123" --name "Mobile App" --limit "30/minute"

# Delete a key
whisper-server keys --keys-file keys.json delete --key "old-key"
```

### 4. Usage Statistics
Monitor how your API is being used:
```bash
whisper-server stats --stats-db data/stats.db --keys-file data/keys.json
```

### 5. Data Retention & Storage
Save every request for auditing or training:
*   Enable by providing `--storage-dir`.
*   The server saves the original `.wav`, the raw transcript, and the LLM-corrected version.
*   Old data is automatically deleted after $N$ days or when the directory exceeds $X$ GB.

## 🧠 Supported Models (Local)

VRAM requirements are approximate for `float16` (GPU standard) vs `int8` (optimized).

| Alias | Full Model ID | VRAM (fp16) | VRAM (int8) | Use Case |
|-------|---------------|-------------|-------------|----------|
| **tiny** | `Systran/faster-whisper-tiny` | 0.6 GB | 0.4 GB | Testing |
| **base** | `Systran/faster-whisper-base` | 0.7 GB | 0.5 GB | Real-time |
| **small** | `Systran/faster-whisper-small` | 1.1 GB | 0.8 GB | General use |
| **medium** | `Systran/faster-whisper-medium` | 2.5 GB | 1.6 GB | Good accuracy |
| **large-v3** | `Systran/faster-whisper-large-v3` | 4.5 GB | 3.0 GB | **Best Accuracy** |
| **turbo** | `deepdml/faster-whisper-large-v3-turbo-ct2` | 2.6 GB | 1.6 GB | **Fast + Accurate** |
| **distil-large-v3** | `Systran/faster-distil-whisper-large-v3` | 2.4 GB | 1.5 GB | Extreme Speed |

## 🎯 OpenAI-Compatible HTTP API

Exposes:
- `POST /v1/audio/transcriptions`
- `POST /v1/audio/translations`

**Additional Form Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vad_filter` | bool | `true` | Enable/disable Silero VAD |
| `response_format` | string | `json` | `json`, `text`, or local-only `verbose_json` |
| `word_timestamps` | bool | `false` | Include per-word timestamps inside `verbose_json` segments |
| `llm_correct` | bool | `false` | Enable LLM post-processing |
| `llm_prompt` | string | - | Custom system prompt for the LLM |

`verbose_json` is supported only by the local `faster-whisper` provider. It cannot be combined with `llm_correct=true` or `speaker:<model>` aliases. `word_timestamps=true` is only valid when `response_format=verbose_json`.

**Example with Auth:**
```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer my-secret-123" \
  -F "file=@sample.wav" \
  -F "model=large-v3" \
  -F "llm_correct=true"
```

**Example `verbose_json` with word timestamps:**
```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@sample.wav" \
  -F "model=large-v3" \
  -F "response_format=verbose_json" \
  -F "word_timestamps=true"
```

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | Required for OpenAI provider |
| `LLM_BASE_URL` | `http://localhost:1234/v1` | URL for the LLM server (LM Studio/OpenAI) |
| `LLM_API_KEY` | `lm-studio` | API key for the LLM server |
| `LLM_MODEL` | `local-model` | Model name to use for LLM correction |
| `VOICES_DIR` | `/data/voices` | Directory for speaker reference files |
| `SPEAKER_THRESHOLD` | `0.80` | Speaker matching threshold (0-1) |
| `RETENTION_DAYS` | `30` | Days to keep logs in storage |
| `RETENTION_MAX_GB` | `10.0` | Maximum size for storage dir |

## 🛠️ CLI Subcommands

- `serve`: Start the STT and Wyoming listeners.
- `stats`: Display usage statistics from the SQLite DB.
- `keys`: Add, remove, or list API keys in the JSON config.

## 📝 License

MIT License
