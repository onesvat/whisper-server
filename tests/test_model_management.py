import pytest
import asyncio
from fastapi.testclient import TestClient
from whisper_server.http_api import create_app
from whisper_server.service import SpeechService
from whisper_server.models import ModelLoader, SttLibrary

@pytest.fixture
def service():
    loader = ModelLoader(
        preferred_stt_library=SttLibrary.AUTO,
        preferred_language="tr",
        download_dir="/tmp/whisper-server-tests",
        local_files_only=True,
        model="tiny",
        compute_type="int8",
        device="cpu",
        beam_size=1,
        cpu_threads=1,
        initial_prompt=None,
        vad_parameters=None,
        provider="local",
    )
    return SpeechService(loader=loader)

@pytest.fixture
def client(service):
    app = create_app(service)
    return TestClient(app)

@pytest.mark.asyncio
async def test_list_models(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) > 0
    
    # Check if 'tiny' is there
    tiny_model = next((m for m in data["data"] if m["id"] == "tiny"), None)
    assert tiny_model is not None
    assert tiny_model["model_id"] == "Systran/faster-whisper-tiny"

@pytest.mark.asyncio
async def test_model_unload(service, client):
    # Load a model first
    model_id, _ = await service._loader.load_transcriber(requested_model="tiny")
    
    # Check it's ready
    response = client.get("/v1/models")
    tiny_model = next((m for m in response.json()["data"] if m["id"] == "tiny"), None)
    assert tiny_model["ready"] is True
    
    # Unload it
    response = client.post("/v1/models/tiny/unload")
    assert response.status_code == 200
    assert response.json()["status"] == "unloaded"
    
    # Check it's no longer ready
    response = client.get("/v1/models")
    tiny_model = next((m for m in response.json()["data"] if m["id"] == "tiny"), None)
    assert tiny_model["ready"] is False

@pytest.mark.asyncio
async def test_auto_unload(service):
    # Warmup model (boot-time / pinned)
    await service.warmup(model="tiny")

    # Load another model later at runtime
    await service._loader.load_transcriber(requested_model="small")

    # Check both are in cache
    info = service.get_models()
    tiny_info = next(m for m in info if m["id"] == "tiny")
    small_info = next(m for m in info if m["id"] == "small")
    assert tiny_info["ready"] is True
    assert small_info["ready"] is True

    # Force auto unload with 0 TTL (only non-pinned idle models are unloaded)
    await service.unload_idle_models(ttl_seconds=-1)

    # Warmup model remains, runtime-loaded model is unloaded
    info = service.get_models()
    tiny_info = next(m for m in info if m["id"] == "tiny")
    small_info = next(m for m in info if m["id"] == "small")
    assert tiny_info["ready"] is True
    assert small_info["ready"] is False
