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

## 🎯 OpenAI-Compatible HTTP API

Exposes:
- `POST /v1/audio/transcriptions`
- `POST /v1/audio/translations`

**Additional Form Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vad_filter` | bool | `true` | Enable/disable Silero VAD |
| `llm_correct` | bool | `false` | Enable LLM post-processing |
| `llm_prompt` | string | - | Custom system prompt for the LLM |

**Example with Auth:**
```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer my-secret-123" \
  -F "file=@sample.wav" \
  -F "model=large-v3" \
  -F "llm_correct=true"
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
