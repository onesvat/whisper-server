"""Tests for compare.py batch transcription script."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import compare


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestCompareScriptCLI:
    """compare.py command-line tests."""

    def test_empty_audio_dir_prints_error(self, tmp_path):
        output_csv = tmp_path / "results.csv"

        result = subprocess.run(
            [
                sys.executable,
                "compare.py",
                "--audio-dir",
                str(tmp_path),
                "--output",
                str(output_csv),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0
        assert f"Error: No .wav files found in {tmp_path}" in result.stdout
        assert not output_csv.exists()

    def test_single_fixture_file_writes_csv(self, tmp_path, monkeypatch):
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        wav_dst = audio_dir / "issai_test_3709.wav"
        wav_dst.write_bytes((FIXTURES_DIR / "issai_test_3709.wav").read_bytes())
        output_csv = tmp_path / "results.csv"

        class FakeModel:
            def __init__(self, model_id, **kwargs):
                self.model_id = model_id
                self.kwargs = kwargs

            def transcribe(self, audio_path, language, beam_size, vad_filter):
                assert Path(audio_path).name == "issai_test_3709.wav"
                assert language == "tr"
                assert beam_size == 1
                assert vad_filter is True
                return [SimpleNamespace(text="merhaba dunya")], None

        fake_module = ModuleType("faster_whisper")
        fake_module.WhisperModel = FakeModel
        monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "compare.py",
                "--audio-dir",
                str(audio_dir),
                "--models",
                "tiny",
                "--compute-types",
                "int8",
                "--beams",
                "1",
                "--output",
                str(output_csv),
                "--download-dir",
                str(tmp_path / "models"),
            ],
        )

        with patch("compare.get_vram_usage", return_value=0), patch(
            "compare.subprocess.run", return_value=SimpleNamespace(returncode=1)
        ):
            compare.main()

        assert output_csv.exists()
        with output_csv.open(newline="", encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))

        assert len(rows) == 1
        assert rows[0]["model"] == "tiny"
        assert rows[0]["file"] == "issai_test_3709.wav"


class TestCompareScriptFunctions:
    """Test compare.py helper functions."""

    def test_calculate_wer_matches_identical_text(self):
        assert compare.calculate_wer("hello world", "hello world") == 0.0

    def test_calculate_wer_handles_missing_reference(self):
        assert compare.calculate_wer("", "anything") == 0.0

    def test_get_vram_usage_returns_zero_when_command_fails(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert compare.get_vram_usage() == 0

    def test_default_model_list_includes_faster_whisper_aliases(self):
        assert "tiny" in compare.DEFAULT_MODELS
        assert "medium" in compare.DEFAULT_MODELS
        assert "turbo" in compare.DEFAULT_MODELS
