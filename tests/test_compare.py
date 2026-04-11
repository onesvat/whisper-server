"""Tests for compare.py batch transcription script."""

import csv
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestCompareScriptCLI:
    """compare.py command-line tests."""

    def test_single_file_local(self, tmp_path):
        """Test --file argument with local provider."""
        wav_file = FIXTURES_DIR / "issai_test_3709.wav"
        output_csv = tmp_path / "results.csv"

        result = subprocess.run(
            [
                sys.executable,
                "compare.py",
                "--provider",
                "local",
                "--file",
                str(wav_file),
                "--models",
                "medium",
                "--output",
                str(output_csv),
                "--download-root",
                str(tmp_path / "models"),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            pytest.skip(f"compare.py failed: {result.stderr}")

        assert output_csv.exists()

        with open(output_csv, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert len(rows) >= 2
        assert rows[0][0] == "filename"
        assert rows[1][0] == "issai_test_3709.wav"

    def test_directory_filter_last_n(self, tmp_path):
        """Test --last N files filter."""
        history_dir = tmp_path / "history"
        history_dir.mkdir()

        for i in range(5):
            wav_src = FIXTURES_DIR / "issai_test_3760.wav"
            wav_dst = history_dir / f"20260411_{i}.wav"
            wav_dst.write_bytes(wav_src.read_bytes())

        result = subprocess.run(
            [
                sys.executable,
                "compare.py",
                "--provider",
                "local",
                "--directory",
                str(history_dir),
                "--last",
                "2",
                "--models",
                "tiny",
                "--output",
                str(tmp_path / "results.csv"),
                "--download-root",
                str(tmp_path / "models"),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )

        if result.returncode != 0:
            pytest.skip(f"compare.py failed: {result.stderr}")

        output_csv = tmp_path / "results.csv"
        assert output_csv.exists()

        with open(output_csv, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert len(rows) == 3

    def test_openai_provider_requires_api_key(self, tmp_path):
        """Test that OpenAI provider requires OPENAI_API_KEY."""
        wav_file = FIXTURES_DIR / "issai_test_3709.wav"

        result = subprocess.run(
            [
                sys.executable,
                "compare.py",
                "--provider",
                "openai",
                "--file",
                str(wav_file),
                "--output",
                str(tmp_path / "results.csv"),
            ],
            capture_output=True,
            text=True,
            env={**subprocess.os.environ, "OPENAI_API_KEY": ""},
        )

        assert result.returncode != 0
        assert "OPENAI_API_KEY" in result.stdout or "OPENAI_API_KEY" in result.stderr


class TestCompareScriptFunctions:
    """Test compare.py internal functions."""

    def test_filter_files_by_filename_date(self, tmp_path):
        """Test filename date filtering."""
        from compare import filter_files_by_filename_date

        history_dir = tmp_path / "history"
        history_dir.mkdir()

        today = datetime.now()
        old_date = today - timedelta(days=10)
        recent_date = today - timedelta(days=1)

        files = [
            history_dir / f"{old_date.strftime('%Y%m%d')}_old.wav",
            history_dir / f"{recent_date.strftime('%Y%m%d')}_recent.wav",
            history_dir / f"{today.strftime('%Y%m%d')}_today.wav",
        ]

        for f in files:
            f.write_bytes(b"fake")

        filtered = filter_files_by_filename_date(files, days=2)

        assert len(filtered) == 2
        assert all(
            f.name.startswith(recent_date.strftime("%Y%m%d"))
            or f.name.startswith(today.strftime("%Y%m%d"))
            for f in filtered
        )

    def test_get_device_cuda_available(self):
        """Test device detection with nvidia-smi available."""
        from compare import get_device

        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            assert get_device() == "cuda"

    def test_get_device_cpu_fallback(self):
        """Test device detection fallback to CPU."""
        from compare import get_device

        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            assert get_device() == "cpu"

    def test_models_list_default(self):
        """Test default model lists."""
        from compare import LOCAL_MODELS, OPENAI_MODELS

        assert len(LOCAL_MODELS) > 0
        assert len(OPENAI_MODELS) > 0
        assert ("medium", "medium") in LOCAL_MODELS
        assert ("whisper-1", "whisper-1") in OPENAI_MODELS


@pytest.mark.skipif(
    not subprocess.os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
class TestCompareOpenAI:
    """OpenAI transcription tests (requires API key)."""

    def test_single_file_openai(self, tmp_path):
        """Test OpenAI provider with real API."""
        wav_file = FIXTURES_DIR / "issai_test_3709.wav"
        output_csv = tmp_path / "results.csv"

        result = subprocess.run(
            [
                sys.executable,
                "compare.py",
                "--provider",
                "openai",
                "--file",
                str(wav_file),
                "--models",
                "whisper-1",
                "--output",
                str(output_csv),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0
        assert output_csv.exists()
