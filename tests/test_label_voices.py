"""Tests for label_voices.py speaker labeling script."""

import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestLabelVoicesFunctions:
    """Test label_voices.py internal functions."""

    def test_get_speakers_empty(self, tmp_path):
        """Test get_speakers with empty directory."""
        from label_voices import get_speakers

        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()

        speakers = get_speakers(voices_dir)
        assert speakers == {}

    def test_get_speakers_single_sample(self, tmp_path):
        """Test get_speakers with single sample per speaker."""
        from label_voices import get_speakers

        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()

        shutil.copy(FIXTURES_DIR / "issai_test_3626.wav", voices_dir / "speaker_a.wav")
        shutil.copy(FIXTURES_DIR / "issai_test_3531.wav", voices_dir / "speaker_b.wav")

        speakers = get_speakers(voices_dir)

        assert speakers == {"speaker_a": 1, "speaker_b": 1}

    def test_get_speakers_multiple_samples(self, tmp_path):
        """Test get_speakers with multiple samples per speaker."""
        from label_voices import get_speakers

        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()

        shutil.copy(FIXTURES_DIR / "issai_test_3626.wav", voices_dir / "speaker_a.wav")
        shutil.copy(
            FIXTURES_DIR / "issai_test_3709.wav", voices_dir / "speaker_a_1.wav"
        )
        shutil.copy(
            FIXTURES_DIR / "issai_test_3760.wav", voices_dir / "speaker_a_2.wav"
        )
        shutil.copy(FIXTURES_DIR / "issai_test_3531.wav", voices_dir / "speaker_b.wav")

        speakers = get_speakers(voices_dir)

        assert speakers == {"speaker_a": 3, "speaker_b": 1}

    def test_get_speakers_ignores_non_wav(self, tmp_path):
        """Test get_speakers ignores non-audio files."""
        from label_voices import get_speakers

        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()

        shutil.copy(FIXTURES_DIR / "issai_test_3626.wav", voices_dir / "speaker.wav")
        (voices_dir / "readme.txt").write_text("test")
        (voices_dir / "data.json").write_text("{}")

        speakers = get_speakers(voices_dir)

        assert speakers == {"speaker": 1}

    def test_play_audio_success(self, tmp_path):
        """Test play_audio with available player."""
        from label_voices import play_audio

        wav_path = tmp_path / "test.wav"
        shutil.copy(FIXTURES_DIR / "issai_test_3760.wav", wav_path)

        with patch("shutil.which") as mock_which:
            mock_which.return_value = "aplay"
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock()

                result = play_audio(wav_path)
                assert result is True
                mock_run.assert_called_once()

    def test_play_audio_no_player(self, tmp_path):
        """Test play_audio when no player available."""
        from label_voices import play_audio

        wav_path = tmp_path / "test.wav"
        shutil.copy(FIXTURES_DIR / "issai_test_3760.wav", wav_path)

        with patch("shutil.which") as mock_which:
            mock_which.return_value = None

            result = play_audio(wav_path)
            assert result is False


class TestLabelVoicesCLI:
    """label_voices.py command-line tests."""

    def test_cli_saved_dir_not_found(self):
        """Test CLI with non-existent saved directory."""
        result = subprocess.run(
            [
                sys.executable,
                "label_voices.py",
                "--saved-dir",
                "/nonexistent/path",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "not found" in result.stdout or "not found" in result.stderr

    def test_cli_no_files(self, tmp_path):
        """Test CLI with empty saved directory."""
        saved_dir = tmp_path / "saved"
        saved_dir.mkdir()

        result = subprocess.run(
            [
                sys.executable,
                "label_voices.py",
                "--saved-dir",
                str(saved_dir),
                "--voices-dir",
                str(tmp_path / "voices"),
            ],
            capture_output=True,
            text=True,
        )

        assert "No audio files found" in result.stdout

    def test_cli_environment_defaults(self, tmp_path):
        """Test CLI uses environment variable defaults."""
        saved_dir = tmp_path / "saved"
        saved_dir.mkdir()
        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()

        shutil.copy(FIXTURES_DIR / "issai_test_3760.wav", saved_dir / "test.wav")

        env = {
            "SAVE_AUDIO_DIR": str(saved_dir),
            "VOICES_DIR": str(voices_dir),
        }

        result = subprocess.run(
            [sys.executable, "label_voices.py"],
            capture_output=True,
            text=True,
            env={**subprocess.os.environ, **env},
        )

        assert "Found 1 audio files" in result.stdout


class TestLabelVoicesFileNaming:
    """Test file naming logic in label_voices."""

    def test_first_sample_no_suffix(self, tmp_path):
        """Test first sample gets name.wav (no suffix)."""
        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()

        shutil.copy(FIXTURES_DIR / "issai_test_3626.wav", voices_dir / "speaker.wav")

        from label_voices import get_speakers

        speakers = get_speakers(voices_dir)

        assert "speaker.wav" in [f.name for f in voices_dir.glob("*.wav")]
        assert speakers["speaker"] == 1

    def test_subsequent_samples_with_suffix(self, tmp_path):
        """Test subsequent samples get name_N.wav suffix."""
        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()

        shutil.copy(FIXTURES_DIR / "issai_test_3626.wav", voices_dir / "speaker.wav")
        shutil.copy(FIXTURES_DIR / "issai_test_3709.wav", voices_dir / "speaker_1.wav")
        shutil.copy(FIXTURES_DIR / "issai_test_3760.wav", voices_dir / "speaker_2.wav")

        from label_voices import get_speakers

        speakers = get_speakers(voices_dir)

        assert speakers["speaker"] == 3
        files = sorted([f.name for f in voices_dir.glob("*.wav")])
        assert files == ["speaker.wav", "speaker_1.wav", "speaker_2.wav"]
