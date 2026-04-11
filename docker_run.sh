#!/usr/bin/env bash
cd /app
python3 -m whisper_server \
    --uri 'tcp://0.0.0.0:10300' \
    --openai-http-host '0.0.0.0' \
    --openai-http-port '8080' \
    --data-dir '/data' "$@"
