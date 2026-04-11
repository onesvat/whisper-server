"""Integration tests with live HTTP and Wyoming servers."""

import json
import socket
import subprocess
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

FIXTURES_DIR = Path(__file__).parent / "fixtures"
EXPECTED_TRANSCRIPTIONS = json.loads(
    (FIXTURES_DIR / "expected_transcriptions.json").read_text()
)

HTTP_PORT = 8080
WYOMING_PORT = 10300


@pytest.fixture(scope="module")
def docker_server():
    """Start docker compose before tests, stop after."""
    compose_file = Path(__file__).parent.parent / "docker-compose.yml"

    subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "up", "-d", "--build"],
        check=True,
        capture_output=True,
    )

    max_wait = 60
    for i in range(max_wait):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            sock.connect(("localhost", WYOMING_PORT))
            sock.close()

            import urllib.request

            urllib.request.urlopen(f"http://localhost:{HTTP_PORT}/healthz", timeout=1)
            break
        except (socket.error, urllib.error.URLError):
            if i < max_wait - 1:
                time.sleep(1)
            else:
                subprocess.run(
                    ["docker", "compose", "-f", str(compose_file), "logs"],
                    capture_output=True,
                )
                pytest.fail("Server did not start within 60 seconds")

    yield

    subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "down"],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def http_client(docker_server):
    """HTTP client for API tests."""
    import requests

    session = requests.Session()
    session.timeout = 30

    def post(url, files=None, data=None):
        return session.post(
            f"http://localhost:{HTTP_PORT}{url}",
            files=files,
            data=data,
        )

    return post


@pytest.fixture
def wyoming_socket(docker_server):
    """Wyoming TCP socket for protocol tests."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("localhost", WYOMING_PORT))
    sock.settimeout(10)
    yield sock
    sock.close()


class TestHTTPIntegration:
    """HTTP API integration tests with real transcription."""

    def test_transcribe_all_fixtures(self, http_client):
        """Test all audio files via HTTP API."""
        for filename, expected in EXPECTED_TRANSCRIPTIONS.items():
            wav_path = FIXTURES_DIR / filename

            with open(wav_path, "rb") as f:
                response = http_client(
                    "/v1/audio/transcriptions",
                    files={"file": (filename, f, "audio/wav")},
                    data={"model": "Systran/faster-whisper-medium"},
                )

            assert response.status_code == 200
            result = response.json()
            assert "text" in result
            assert expected[:20] in result["text"], (
                f"{filename}: expected '{expected[:20]}...' in '{result['text']}'"
            )

    def test_model_override(self, http_client):
        """Test requesting different model via API."""
        wav_path = FIXTURES_DIR / "issai_test_3760.wav"

        with open(wav_path, "rb") as f:
            response = http_client(
                "/v1/audio/transcriptions",
                files={"file": ("issai_test_3760.wav", f, "audio/wav")},
                data={"model": "Systran/faster-whisper-small"},
            )

        assert response.status_code == 200
        result = response.json()
        expected = EXPECTED_TRANSCRIPTIONS["issai_test_3760.wav"]
        assert expected[:20] in result["text"]

    def test_response_format_text(self, http_client):
        """Test text response format."""
        wav_path = FIXTURES_DIR / "issai_test_3709.wav"

        with open(wav_path, "rb") as f:
            response = http_client(
                "/v1/audio/transcriptions",
                files={"file": ("issai_test_3709.wav", f, "audio/wav")},
                data={
                    "model": "Systran/faster-whisper-medium",
                    "response_format": "text",
                },
            )

        assert response.status_code == 200
        text = response.text
        expected = EXPECTED_TRANSCRIPTIONS["issai_test_3709.wav"]
        assert expected[:20] in text


class TestWyomingIntegration:
    """Wyoming TCP protocol integration tests."""

    def test_describe_info_exchange(self, wyoming_socket):
        """Test Describe -> Info exchange."""
        from wyoming.info import Describe
        from wyoming.event import write_event

        describe_event = Describe().event()
        write_event(describe_event, wyoming_socket.makefile("wb"))

        response = _read_event(wyoming_socket)
        assert "info" in response.type

    def test_transcribe_audio_chunk(self, wyoming_socket):
        """Test full transcription flow via Wyoming."""
        from wyoming.audio import AudioChunk, AudioStop
        from wyoming.event import write_event, read_event

        wav_path = FIXTURES_DIR / "issai_test_3709.wav"
        wav_data = wav_path.read_bytes()
        pcm_data = _wav_to_pcm16(wav_data)

        chunk = AudioChunk(rate=16000, width=2, channels=1, audio=pcm_data)
        write_event(chunk.event(), wyoming_socket.makefile("wb"))

        stop_event = AudioStop().event()
        write_event(stop_event, wyoming_socket.makefile("wb"))

        response = read_event(wyoming_socket.makefile("rb"))
        assert response is not None

        from wyoming.asr import Transcript

        if Transcript.is_type(response.type):
            transcript = Transcript.from_event(response)
            expected = EXPECTED_TRANSCRIPTIONS["issai_test_3709.wav"]
            assert expected[:20] in transcript.text


def _wav_to_pcm16(wav_bytes):
    """Convert WAV bytes to raw PCM16."""
    import io
    import wave

    with io.BytesIO(wav_bytes) as buffer:
        with wave.open(buffer, "rb") as wav:
            return wav.readframes(wav.getnframes())


def _read_event(sock):
    """Read Wyoming event from socket."""
    from wyoming.event import read_event

    return read_event(sock.makefile("rb"))
