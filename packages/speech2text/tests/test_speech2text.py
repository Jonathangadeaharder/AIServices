import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from speech2text.models import Speech2TextRequest, Speech2TextResponse


@pytest.fixture
def dummy_request(tmp_path):
    return Speech2TextRequest(
        audio_path=str(tmp_path / "test_audio.wav"),
        model_name="mlx-community/whisper-large-v3-mlx",
    )


class TestModelValidation:
    """Test Pydantic model validation for Speech2Text models."""

    def test_request_defaults(self):
        """Test that default values are applied correctly."""
        req = Speech2TextRequest(audio_path="/tmp/audio.wav")
        assert req.audio_path == "/tmp/audio.wav"
        assert req.model_name == "mlx-community/whisper-large-v3-mlx"
        assert req.language is None

    def test_request_custom_values(self):
        """Test that custom values are accepted."""
        req = Speech2TextRequest(
            audio_path="/tmp/audio.wav",
            model_name="mlx-community/whisper-small-mlx",
            language="en",
        )
        assert req.model_name == "mlx-community/whisper-small-mlx"
        assert req.language == "en"

    def test_request_requires_audio_path(self):
        """Test that audio_path is required."""
        with pytest.raises(ValidationError):
            Speech2TextRequest()

    def test_response_model(self):
        """Test response model construction."""
        resp = Speech2TextResponse(
            text="Hello world",
            metadata={"provider": "mlx-whisper", "language": "en"},
        )
        assert resp.text == "Hello world"
        assert resp.metadata["provider"] == "mlx-whisper"

    def test_response_default_metadata(self):
        """Test that metadata defaults to empty dict."""
        resp = Speech2TextResponse(text="test")
        assert resp.metadata == {}


@patch("speech2text.providers.mlx.mlx_whisper")
def test_mlx_provider_mocked(mock_mlx_whisper, dummy_request, tmp_path):
    """Test the MLXWhisperProvider with mocked mlx_whisper."""
    from speech2text.providers.mlx import MLXWhisperProvider

    # Setup mock
    mock_mlx_whisper.transcribe.return_value = {
        "text": "  Hello, this is a test transcription.  ",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "Hello, this is a test transcription."},
        ],
    }

    provider = MLXWhisperProvider()
    out_file = tmp_path / "transcript.txt"

    response = provider.generate(dummy_request, str(out_file))

    # Verify transcription result
    assert response.text == "Hello, this is a test transcription."
    assert response.metadata["provider"] == "mlx-whisper"
    assert response.metadata["language"] == "en"
    assert response.metadata["model"] == "mlx-community/whisper-large-v3-mlx"
    assert len(response.metadata["segments"]) == 1

    # Verify mlx_whisper was called correctly
    mock_mlx_whisper.transcribe.assert_called_once_with(
        dummy_request.audio_path,
        path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
        language=None,
    )

    # Verify output file was written
    assert out_file.exists()
    assert out_file.read_text() == "Hello, this is a test transcription."


@patch("speech2text.providers.mlx.mlx_whisper")
def test_mlx_provider_no_output_file(mock_mlx_whisper, dummy_request):
    """Test provider works without writing to an output file."""
    from speech2text.providers.mlx import MLXWhisperProvider

    mock_mlx_whisper.transcribe.return_value = {
        "text": "No file output",
        "language": "en",
        "segments": [],
    }

    provider = MLXWhisperProvider()
    response = provider.generate(dummy_request, None)

    assert response.text == "No file output"


@patch("speech2text.providers.mlx.mlx_whisper")
def test_mlx_provider_empty_transcription(mock_mlx_whisper, dummy_request):
    """Test provider handles empty transcription gracefully."""
    from speech2text.providers.mlx import MLXWhisperProvider

    mock_mlx_whisper.transcribe.return_value = {
        "text": "   ",
        "language": None,
        "segments": [],
    }

    provider = MLXWhisperProvider()
    response = provider.generate(dummy_request, None)

    assert response.text == ""
    assert response.metadata["language"] is None


@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS") != "1",
    reason="Requires RUN_INTEGRATION_TESTS=1",
)
def test_mlx_whisper_integration(tmp_path):
    """Integration test requiring mlx-whisper and a real audio file."""
    audio_path = os.environ.get("SPEECH2TEXT_TEST_AUDIO")
    if not audio_path:
        pytest.skip("Set SPEECH2TEXT_TEST_AUDIO to a real audio file path")

    from speech2text.providers.mlx import MLXWhisperProvider

    request = Speech2TextRequest(
        audio_path=audio_path,
        model_name="mlx-community/whisper-large-v3-mlx",
    )

    provider = MLXWhisperProvider()
    out_file = tmp_path / "transcript.txt"

    response = provider.generate(request, str(out_file))
    assert isinstance(response.text, str)
    assert out_file.exists()
