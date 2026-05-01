import os
from unittest.mock import MagicMock, patch

import pytest
from text2speech.models import Text2SpeechRequest


def test_model_validation():
    req = Text2SpeechRequest(text="Hello", emotion="happy")
    assert req.text == "Hello"
    assert req.emotion == "happy"


def test_model_defaults():
    req = Text2SpeechRequest(text="test")
    assert req.voice_id is None
    assert req.emotion is None
    assert req.reference_audio is None


def test_reference_fields():
    req = Text2SpeechRequest(
        text="Hello",
        reference_audio="/tmp/ref.wav",
        reference_text="Reference text",
    )
    assert req.reference_audio == "/tmp/ref.wav"
    assert req.reference_text == "Reference text"


@patch("urllib.request.urlopen")
@patch("urllib.request.Request")
def test_fish_provider_api(mock_request, mock_urlopen, tmp_path):
    from text2speech.providers.fish import FishSpeechProvider

    mock_response = MagicMock()
    mock_response.read.return_value = b"fake_audio_data"
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    provider = FishSpeechProvider(api_url="http://localhost:8090")
    out_file = tmp_path / "out.wav"

    request = Text2SpeechRequest(text="Hello", voice_id="char_1", emotion="happy")

    response = provider.generate(request, str(out_file))

    assert response.output_path == str(out_file)
    assert response.metadata["provider"] == "fish-speech-api"
    assert out_file.exists()
    assert out_file.read_bytes() == b"fake_audio_data"

    assert provider._build_fish_text(request) == "[happy] Hello"


@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS") != "1",
    reason="Requires RUN_INTEGRATION_TESTS=1",
)
def test_fish_mlx_integration(tmp_path):
    from text2speech.providers.fish_mlx import FishMLXProvider

    provider = FishMLXProvider()
    request = Text2SpeechRequest(text="Hello, this is a test.")
    out_file = tmp_path / "test_audio.wav"

    response = provider.generate(request, str(out_file))
    assert response.output_path == str(out_file)
    assert response.metadata["provider"] == "fish-s2-pro-mlx"
