"""Speaker recognition tests using issai audio fixtures."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from whisper_server.audio import normalize_audio_file
from whisper_server.speaker_recognition import SpeakerConfig, SpeakerRecognizer

pytestmark = pytest.mark.integration

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def tmp_voices_dir(tmp_path):
    """Temporary voices directory for speaker registration."""
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    return voices_dir


@pytest.fixture
def audio_samples():
    """Load audio samples from fixtures."""
    samples = {}
    for filename in [
        "issai_test_3531.wav",
        "issai_test_3626.wav",
        "issai_test_3709.wav",
    ]:
        wav_path = FIXTURES_DIR / filename
        normalized = normalize_audio_file(wav_path)
        samples[filename] = normalized.samples
    return samples


class TestSpeakerRecognition:
    """Speaker detection tests with 3626/3709 same speaker."""

    def test_same_speaker_recognized(self, tmp_voices_dir, audio_samples):
        """Register 3626 as speaker 'test_speaker', test 3709 - should match."""
        shutil.copy(
            FIXTURES_DIR / "issai_test_3626.wav",
            tmp_voices_dir / "test_speaker.wav",
        )

        config = SpeakerConfig(
            voices_dir=tmp_voices_dir,
            threshold=0.80,
            scan_interval_s=0,
        )
        recognizer = SpeakerRecognizer(config)

        result = recognizer.identify(audio_samples["issai_test_3709.wav"])

        assert result is not None
        assert result[0] == "test_speaker"
        assert result[1] >= 0.80

    def test_different_speaker_rejected(self, tmp_voices_dir, audio_samples):
        """Register 3626 as speaker, test 3531 - should NOT match."""
        shutil.copy(
            FIXTURES_DIR / "issai_test_3626.wav",
            tmp_voices_dir / "test_speaker.wav",
        )

        config = SpeakerConfig(
            voices_dir=tmp_voices_dir,
            threshold=0.80,
            scan_interval_s=0,
        )
        recognizer = SpeakerRecognizer(config)

        result = recognizer.identify(audio_samples["issai_test_3531.wav"])

        assert result is None or result[1] < 0.80

    def test_multiple_samples_per_speaker(self, tmp_voices_dir, audio_samples):
        """Register multiple samples for same speaker, test averaging."""
        shutil.copy(
            FIXTURES_DIR / "issai_test_3626.wav",
            tmp_voices_dir / "test_speaker.wav",
        )
        shutil.copy(
            FIXTURES_DIR / "issai_test_3709.wav",
            tmp_voices_dir / "test_speaker_1.wav",
        )

        config = SpeakerConfig(
            voices_dir=tmp_voices_dir,
            threshold=0.80,
            scan_interval_s=0,
        )
        recognizer = SpeakerRecognizer(config)

        result = recognizer.identify(audio_samples["issai_test_3709.wav"])

        assert result is not None
        assert result[0] == "test_speaker"

    def test_threshold_boundary(self, tmp_voices_dir, audio_samples):
        """Test threshold boundary - lower threshold should match."""
        shutil.copy(
            FIXTURES_DIR / "issai_test_3626.wav",
            tmp_voices_dir / "test_speaker.wav",
        )

        high_threshold_config = SpeakerConfig(
            voices_dir=tmp_voices_dir,
            threshold=0.99,
            scan_interval_s=0,
        )
        high_recognizer = SpeakerRecognizer(high_threshold_config)
        high_result = high_recognizer.identify(audio_samples["issai_test_3709.wav"])

        low_threshold_config = SpeakerConfig(
            voices_dir=tmp_voices_dir,
            threshold=0.50,
            scan_interval_s=0,
        )
        low_recognizer = SpeakerRecognizer(low_threshold_config)
        low_result = low_recognizer.identify(audio_samples["issai_test_3709.wav"])

        assert low_result is not None

    def test_no_voices_registered(self, tmp_voices_dir, audio_samples):
        """Test with empty voices directory."""
        config = SpeakerConfig(
            voices_dir=tmp_voices_dir,
            threshold=0.80,
            scan_interval_s=0,
        )
        recognizer = SpeakerRecognizer(config)

        result = recognizer.identify(audio_samples["issai_test_3709.wav"])

        assert result is None

    def test_dynamic_refresh(self, tmp_voices_dir, audio_samples):
        """Test that recognizer refreshes when new voices added."""
        config = SpeakerConfig(
            voices_dir=tmp_voices_dir,
            threshold=0.80,
            scan_interval_s=0,
        )
        recognizer = SpeakerRecognizer(config)

        result_before = recognizer.identify(audio_samples["issai_test_3709.wav"])
        assert result_before is None

        shutil.copy(
            FIXTURES_DIR / "issai_test_3626.wav",
            tmp_voices_dir / "test_speaker.wav",
        )

        recognizer.refresh()

        result_after = recognizer.identify(audio_samples["issai_test_3709.wav"])
        assert result_after is not None
