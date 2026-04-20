import pytest

from whisper_server.const import SttLibrary
from whisper_server.models import ModelLoader, normalize_model_name


class DummyTranscriber:
    def __init__(self, model_id: str, **kwargs) -> None:
        self.model_id = model_id
        self.kwargs = kwargs


@pytest.mark.asyncio
async def test_model_loader_reuses_request_scoped_override(monkeypatch):
    created_models: list[str] = []

    def fake_transcriber(model_id, **kwargs):
        created_models.append(model_id)
        return DummyTranscriber(model_id, **kwargs)

    monkeypatch.setattr(
        "whisper_server.models.FasterWhisperTranscriber", fake_transcriber
    )

    loader = ModelLoader(
        preferred_stt_library=SttLibrary.AUTO,
        preferred_language="tr",
        download_dir="/tmp/whisper-server-tests",
        local_files_only=True,
        model="base",
        compute_type="int8",
        device="cpu",
        beam_size=5,
        cpu_threads=1,
        initial_prompt=None,
        vad_parameters=None,
        provider="local",
    )

    model_a, transcriber_a = await loader.load_transcriber(
        language="tr", requested_model="large-v3"
    )
    model_b, transcriber_b = await loader.load_transcriber(
        language="tr", requested_model="large-v3"
    )

    assert model_a == "Systran/faster-whisper-large-v3"
    assert model_b == "Systran/faster-whisper-large-v3"
    assert transcriber_a is transcriber_b
    assert created_models == ["Systran/faster-whisper-large-v3"]


def test_normalize_model_name_expands_short_int8_alias():
    assert normalize_model_name("base") == "Systran/faster-whisper-base"
    assert normalize_model_name("large-v3") == "Systran/faster-whisper-large-v3"
    assert normalize_model_name("medium") == "Systran/faster-whisper-medium"
    assert (
        normalize_model_name("turbo") == "deepdml/faster-whisper-large-v3-turbo-ct2"
    )
    assert (
        normalize_model_name("Systran/faster-whisper-medium")
        == "Systran/faster-whisper-medium"
    )
