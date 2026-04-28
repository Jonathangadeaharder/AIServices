
from text2speech.models import Text2SpeechRequest


def test_model_validation():
    """Test that model validation works."""
    req = Text2SpeechRequest(text="Hello")
    assert req.text == "Hello"


def test_model_defaults():
    """Test default values."""
    req = Text2SpeechRequest(text="test")
    assert req.voice_id is None


def test_fish_provider_scaffold(tmp_path):
    """Test the FishSpeechProvider scaffold."""
    from text2speech.providers.fish import FishSpeechProvider

    provider = FishSpeechProvider()
    out_file = tmp_path / "out.wav"

    response = provider.generate(Text2SpeechRequest(text="test"), str(out_file))
    assert response.output_path == str(out_file)
    assert response.metadata["provider"] == "fish-speech"
