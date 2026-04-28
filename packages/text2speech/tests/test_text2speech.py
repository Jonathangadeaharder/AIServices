from unittest.mock import MagicMock, patch

from text2speech.models import Text2SpeechRequest


def test_model_validation():
    """Test that model validation works."""
    req = Text2SpeechRequest(text="Hello", emotion="happy")
    assert req.text == "Hello"
    assert req.emotion == "happy"


def test_model_defaults():
    """Test default values."""
    req = Text2SpeechRequest(text="test")
    assert req.voice_id is None
    assert req.emotion is None


@patch("urllib.request.urlopen")
@patch("urllib.request.Request")
def test_fish_provider_api(mock_request, mock_urlopen, tmp_path):
    """Test the FishSpeechProvider with mocked API call."""
    from text2speech.providers.fish import FishSpeechProvider

    # Setup mock
    mock_response = MagicMock()
    mock_response.read.return_value = b"fake_audio_data"
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    provider = FishSpeechProvider(api_url="http://localhost:8090")
    out_file = tmp_path / "out.wav"

    request = Text2SpeechRequest(
        text="Hello", 
        voice_id="char_1",
        emotion="happy"
    )
    
    response = provider.generate(request, str(out_file))
    
    assert response.output_path == str(out_file)
    assert response.metadata["provider"] == "fish-speech-api"
    
    # Verify file content
    assert out_file.exists()
    assert out_file.read_bytes() == b"fake_audio_data"
    
    # Verify build_fish_text logic
    assert provider._build_fish_text(request) == "(happy) Hello"
