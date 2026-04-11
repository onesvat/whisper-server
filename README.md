# whisper-server

> Shared Wyoming TCP and OpenAI-compatible HTTP speech-to-text service with optional speaker tagging

[![Docker Hub](https://img.shields.io/docker/pulls/onesvat/whisper-server.svg)](https://hub.docker.com/r/onesvat/whisper-server)

## ✨ Features

- **Two Protocols, One Process**
  - Wyoming TCP ASR on `10300`
  - OpenAI-compatible HTTP STT on `8080`
  - Shared model cache across both listeners

- **Two Transcription Providers**
  - 🏠 **Local**: Faster-Whisper with CTranslate2
  - ☁️ **OpenAI**: outbound OpenAI transcription and translation APIs

- **Optional Speaker Recognition**
  - Speaker tagging is request-scoped via `speaker:<model>`
  - Tags whole utterances as `[onur] Hello, how are you?`

- **Flexible Model Comparison**
  - Batch compare different models on your audio files
  - Supports both local and OpenAI models
  - Export results to CSV for analysis

- **GPU Support**
  - NVIDIA CUDA 12.4 support
  - Automatic device detection
  - Float16/Int8 quantization

## 🚀 Quick Start

### Option 1: Local Transcription (GPU Required)

Best for: Privacy, no API costs, unlimited usage

```yaml
# docker-compose.yml
services:
  whisper:
    image: onesvat/whisper-server:gpu
    environment:
      - TZ=Europe/Istanbul
    volumes:
      - ./data:/data
    ports:
      - "10300:10300"
      - "8080:8080"
    command: [
      "--model", "large-v3",
      "--language", "tr",
      "--device", "cuda",
      "--compute-type", "float16"
    ]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

```bash
docker compose up -d
```

### Option 2: OpenAI API (No GPU Required)

Best for: Quick setup, no GPU, pay-per-use

```yaml
# docker-compose.yml
services:
  whisper:
    image: onesvat/whisper-server:gpu
    environment:
      - TZ=Europe/Istanbul
      - OPENAI_API_KEY=sk-your-api-key-here
    volumes:
      - ./data:/data
    ports:
      - "10300:10300"
      - "8080:8080"
    command: [
      "--provider", "openai",
      "--model", "gpt-4o-transcribe"
    ]
```

```bash
docker compose up -d
```

## 📖 How It Works

### 1. Speech-to-Text

The service accepts audio through Wyoming or OpenAI-compatible HTTP and returns transcriptions:

```
Wyoming TCP or HTTP Upload → Shared Service Layer → Text Output
                                     ↓
                            [Local or OpenAI]
```

### 2. Speaker Recognition

When the requested model is prefixed with `speaker:`, the audio is analyzed to identify the speaker:

```
Audio Input → Speaker Matching → [name] Transcribed text
                ↓
        Reference voices in /data/voices/
```

### 3. OpenAI-Compatible HTTP

`whisper-server` exposes:

- `POST /v1/audio/transcriptions`
- `POST /v1/audio/translations`
- `GET /healthz`

Example:

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@sample.wav" \
  -F "model=speaker:large-v3" \
  -F "response_format=json"
```

### 4. Model Comparison Tool

The `compare.py` script helps you test different models:

```
Audio Files → Multiple Models → CSV Comparison
```

## 🎯 Use Cases

### Home Assistant Voice Assistant

Integrate with Home Assistant for voice control:

```yaml
# Home Assistant configuration.yaml
wyoming:
  - host: whisper
    port: 10300
```

### Voice Transcription Service

Use as a standalone transcription microservice:

```bash
# Send audio via Wyoming protocol
# Receive transcribed text
```

### Model Benchmarking

Compare transcription accuracy across models:

```bash
python3 compare.py --provider openai --last 10
```

## 🔧 Configuration

### Provider Selection

| Provider | CLI Argument | Environment Variable | GPU Required |
|----------|--------------|---------------------|--------------|
| Local (default) | `--provider local` | None | ✅ Recommended |
| OpenAI | `--provider openai` | `OPENAI_API_KEY` | ❌ No |

### Available Models

#### Local Models (Faster-Whisper)

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| tiny | ⚡⚡⚡⚡⚡ | ⭐⭐ | Testing |
| base | ⚡⚡⚡⚡ | ⭐⭐⭐ | Real-time |
| small | ⚡⚡⚡ | ⭐⭐⭐⭐ | General use |
| medium | ⚡⚡ | ⭐⭐⭐⭐ | Good accuracy |
| large-v2 | ⚡ | ⭐⭐⭐⭐⭐ | High accuracy |
| **large-v3** | ⚡ | ⭐⭐⭐⭐⭐ | **Best accuracy** (recommended) |
| turbo | ⚡⚡⚡ | ⭐⭐⭐⭐ | Fast + accurate |
| selimc | ⚡ | ⭐⭐⭐⭐ | Turkish-optimized |

#### OpenAI Models

| Model | Cost | Quality | Use Case |
|-------|------|---------|----------|
| whisper-1 | $ | Good | Standard Whisper |
| gpt-4o-mini-transcribe | $$ | Better | Fast, lower cost |
| **gpt-4o-transcribe** | $$$ | Best | **Highest quality** (recommended) |
| gpt-4o-transcribe-diarize | $$$ | Best + Diarization | Multi-speaker |

### Speaker Recognition Setup

1. **Create voice reference files:**
   ```bash
   mkdir -p data/voices
   # Add reference audio (5-15 seconds per speaker)
   cp ~/recordings/onur-sample.wav data/voices/onur.wav
   cp ~/recordings/gamze-sample.wav data/voices/gamze.wav
   ```

2. **Request speaker tagging with the model alias:**
   ```
   curl -X POST http://localhost:8080/v1/audio/transcriptions \
     -F "file=@sample.wav" \
     -F "model=speaker:large-v3"
   ```

3. **Adjust threshold if needed:**
   ```yaml
   environment:
     - SPEAKER_THRESHOLD=0.85  # Lower = more matches
   ```

## 📊 Model Comparison Tool

### Basic Usage

```bash
# Compare last 10 files with default models
python3 compare.py --last 10

# Compare specific OpenAI models
python3 compare.py --provider openai --models whisper-1,gpt-4o-transcribe --last 5

# Compare last 2 days
python3 compare.py --last-days 2
```

### With Docker

```bash
# Local models
docker run --rm --gpus all \
  --entrypoint python3 \
  -v ./data:/data \
  onesvat/whisper-server:gpu \
  /app/compare.py --last 10

# OpenAI models
docker run --rm \
  --entrypoint python3 \
  -v ./data:/data \
  -e OPENAI_API_KEY=sk-xxx \
  onesvat/whisper-server:gpu \
  /app/compare.py --provider openai --last 10
```

### Output

Results saved to CSV:

| filename | tiny | base | medium | large-v3 |
|----------|------|------|--------|----------|
| audio1.wav | Nefesin? | Nefesim | Nasılsın? | Nasılsın? |
| audio2.wav | Bugün... | Bugün... | Bugün nasılsın? | Bugün nasılsın? |

## 🐳 Docker Images

### Pre-built Images

```bash
# Pull from Docker Hub
docker pull onesvat/whisper-server:gpu
docker pull onesvat/whisper-server:cpu
```

### Build Locally

```bash
# GPU version
docker build -f Dockerfile.gpu -t whisper-server:gpu .

# CPU version
docker build -f Dockerfile.cpu -t whisper-server:cpu .
```

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | **Required for OpenAI provider** |
| `VOICES_DIR` | `/data/voices` | Directory for speaker reference files |
| `SPEAKER_THRESHOLD` | `0.80` | Speaker matching threshold (0-1) |
| `VOICE_SCAN_INTERVAL` | `10` | Seconds between voice file scans |
| `SAVE_AUDIO_DIR` | - | Directory to save transcribed audio |
| `TZ` | `UTC` | Timezone for timestamps |

## 🔍 Advanced Examples

### Production Setup with GPU

```yaml
services:
  whisper:
    image: onesvat/whisper-server:gpu
    container_name: whisper
    restart: unless-stopped
    environment:
      - TZ=Europe/Istanbul
      - VOICES_DIR=/data/voices
      - SPEAKER_THRESHOLD=0.85
      - SAVE_AUDIO_DIR=/data/history
    volumes:
      - ./data/whisper:/data
    ports:
      - "10300:10300"
      - "8080:8080"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: [
      "--model", "large-v3",
      "--beam-size", "5",
      "--language", "tr",
      "--device", "cuda",
      "--compute-type", "float16"
    ]
```

### OpenAI with Fallback

```yaml
services:
  whisper:
    image: onesvat/whisper-server:gpu
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    command: [
      "--provider", "openai",
      "--model", "gpt-4o-transcribe",
      "--language", "tr"
    ]
```

## 🛠️ Development

### Project Structure

```
whisper-server/
├── compare.py                    # Model comparison tool
├── label_voices.py              # Voice labeling utility
├── Dockerfile.gpu               # GPU Docker image
├── Dockerfile.cpu               # CPU Docker image
└── whisper_server/
    ├── __main__.py              # Entry point
    ├── service.py               # Shared request orchestration
    ├── models.py                # Model loader/cache
    ├── const.py                 # Constants
    ├── wyoming_handler.py       # Wyoming protocol handler
    ├── http_api.py              # OpenAI-compatible HTTP API
    ├── speaker_recognition.py   # Speaker identification
    ├── faster_whisper_handler.py # Local transcription
    └── openai_transcriber.py    # OpenAI transcription
```

### Running Tests

```bash
pytest tests/
```

## ❓ FAQ

**Q: Which provider should I use?**

A: Use **local** if you have a GPU and want low-latency reusable models. Use **OpenAI** if you don't have a GPU or want outbound API-backed transcription and translation.

**Q: Which model is best?**

A: For local, use **large-v3** for best accuracy or **turbo** for speed. For OpenAI, use **gpt-4o-transcribe**.

**Q: How accurate is speaker recognition?**

A: Accuracy depends on voice sample quality. Use 10-15 second samples with clear speech, and request speaker tagging with `speaker:<model>`. Lower `SPEAKER_THRESHOLD` if matches are missed.

**Q: Can I use this without GPU?**

A: Yes! Use `--provider openai` or run local models on CPU (slower).

**Q: How much GPU memory do I need?**

A: 
- tiny/base: 1GB
- medium: 2GB
- large-v3: 4GB
- turbo: 2GB

## 📝 License

MIT License

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines.

## 📚 Related Projects

- [Wyoming Protocol](https://github.com/rhasspy/wyoming)
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper)
- [OpenAI Whisper](https://github.com/openai/whisper)
