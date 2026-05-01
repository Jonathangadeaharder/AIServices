import importlib.util
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

_HAS_ORMSGPACK = importlib.util.find_spec("ormsgpack") is not None


def test_model_validation():
    from text2speech.models import Text2SpeechRequest

    req = Text2SpeechRequest(text="Hello", emotion="happy")
    assert req.text == "Hello"
    assert req.emotion == "happy"


def test_model_defaults():
    from text2speech.models import Text2SpeechRequest

    req = Text2SpeechRequest(text="test")
    assert req.voice_id is None
    assert req.emotion is None
    assert req.reference_audio is None


def test_reference_fields():
    from text2speech.models import Text2SpeechRequest

    req = Text2SpeechRequest(
        text="Hello",
        reference_audio="/tmp/ref.wav",
        reference_text="Reference text",
    )
    assert req.reference_audio == "/tmp/ref.wav"
    assert req.reference_text == "Reference text"


@pytest.mark.skipif(not _HAS_ORMSGPACK, reason="ormsgpack not installed")
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

    from text2speech.models import Text2SpeechRequest

    request = Text2SpeechRequest(text="Hello", voice_id="char_1", emotion="happy")

    response = provider.generate(request, str(out_file))

    assert response.output_path == str(out_file)
    assert response.metadata["provider"] == "fish-speech-api"
    assert out_file.exists()
    assert out_file.read_bytes() == b"fake_audio_data"

    assert provider._build_fish_text(request) == "[happy] Hello"


def test_fish_provider_api_mocked(tmp_path):
    mock_ormsgpack = MagicMock()
    mock_ormsgpack.packb.return_value = b"packed_data"
    sys.modules["ormsgpack"] = mock_ormsgpack

    from text2speech.providers.fish import FishSpeechProvider

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = b"fake_audio_data"
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        provider = FishSpeechProvider(api_url="http://localhost:8090")
        out_file = tmp_path / "out.wav"

        from text2speech.models import Text2SpeechRequest

        request = Text2SpeechRequest(text="Hello", voice_id="char_1", emotion="happy")

        response = provider.generate(request, str(out_file))

        assert response.output_path == str(out_file)
        assert response.metadata["provider"] == "fish-speech-api"
        assert out_file.exists()


def test_fish_provider_build_text():
    from text2speech.providers.fish import FishSpeechProvider
    from text2speech.models import Text2SpeechRequest

    provider = FishSpeechProvider()

    req = Text2SpeechRequest(text="Hello")
    assert provider._build_fish_text(req) == "Hello"

    req_emotion = Text2SpeechRequest(text="Hello", emotion="happy")
    assert provider._build_fish_text(req_emotion) == "[happy] Hello"

    req_tone = Text2SpeechRequest(text="Hello", tone="calm")
    assert provider._build_fish_text(req_tone) == "[calm] Hello"

    req_effect = Text2SpeechRequest(text="Hello", effect="echo")
    assert provider._build_fish_text(req_effect) == "[echo] Hello"


@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS") != "1",
    reason="Requires RUN_INTEGRATION_TESTS=1",
)
def test_fish_mlx_integration(tmp_path):
    from text2speech.providers.fish_mlx import FishMLXProvider

    provider = FishMLXProvider()
    from text2speech.models import Text2SpeechRequest

    request = Text2SpeechRequest(text="Hello, this is a test.")
    out_file = tmp_path / "test_audio.wav"

    response = provider.generate(request, str(out_file))
    assert response.output_path == str(out_file)
    assert response.metadata["provider"] == "fish-s2-pro-mlx"


def test_fish_mlx_build_text():
    sys.modules["mlx_audio"] = MagicMock()
    sys.modules["mlx_audio.tts"] = MagicMock()
    sys.modules["mlx_audio.tts.utils"] = MagicMock()
    sys.modules["mlx_audio.tts.generate"] = MagicMock()

    from text2speech.models import Text2SpeechRequest
    from text2speech.providers.fish_mlx import FishMLXProvider

    provider = FishMLXProvider()

    req = Text2SpeechRequest(text="Hello")
    assert provider._build_text(req) == "Hello"

    req_emotion = Text2SpeechRequest(text="Hello", emotion="happy")
    assert provider._build_text(req_emotion) == "[happy] Hello"

    req_tone = Text2SpeechRequest(text="Hello", tone="calm")
    assert provider._build_text(req_tone) == "[calm] Hello"

    req_effect = Text2SpeechRequest(text="Hello", effect="echo")
    assert provider._build_text(req_effect) == "[echo] Hello"

    req_all = Text2SpeechRequest(text="Hello", emotion="happy", tone="calm", effect="echo")
    assert provider._build_text(req_all) == "[happy][calm][echo] Hello"


def test_fish_mlx_generate_mocked(tmp_path):
    mock_load_model = MagicMock()
    mock_generate_audio = MagicMock()

    sys.modules["mlx_audio"] = MagicMock()
    sys.modules["mlx_audio.tts"] = MagicMock()
    sys.modules["mlx_audio.tts.utils"] = MagicMock()
    sys.modules["mlx_audio.tts.utils"].load_model = mock_load_model
    sys.modules["mlx_audio.tts.generate"] = MagicMock()
    sys.modules["mlx_audio.tts.generate"].generate_audio = mock_generate_audio

    from text2speech.models import Text2SpeechRequest
    from text2speech.providers.fish_mlx import FishMLXProvider

    provider = FishMLXProvider()
    out_file = tmp_path / "out.wav"
    request = Text2SpeechRequest(text="Hello", voice_id="test_voice")

    response = provider.generate(request, str(out_file))

    assert response.output_path == str(out_file)
    assert response.metadata["provider"] == "fish-s2-pro-mlx"
    assert response.metadata["voice_id"] == "test_voice"
    mock_load_model.assert_called_once()
    mock_generate_audio.assert_called_once()


def test_fish_mlx_generate_no_output_path():
    mock_load_model = MagicMock()
    mock_generate_audio = MagicMock()

    sys.modules["mlx_audio"] = MagicMock()
    sys.modules["mlx_audio.tts"] = MagicMock()
    sys.modules["mlx_audio.tts.utils"] = MagicMock()
    sys.modules["mlx_audio.tts.utils"].load_model = mock_load_model
    sys.modules["mlx_audio.tts.generate"] = MagicMock()
    sys.modules["mlx_audio.tts.generate"].generate_audio = mock_generate_audio

    from text2speech.models import Text2SpeechRequest
    from text2speech.providers.fish_mlx import FishMLXProvider

    provider = FishMLXProvider()
    request = Text2SpeechRequest(text="Hello")

    response = provider.generate(request)

    assert response.output_path.endswith(".wav")
    assert response.metadata["provider"] == "fish-s2-pro-mlx"


def test_fish_mlx_generate_with_reference_audio(tmp_path):
    mock_load_model = MagicMock()
    mock_generate_audio = MagicMock()

    sys.modules["mlx_audio"] = MagicMock()
    sys.modules["mlx_audio.tts"] = MagicMock()
    sys.modules["mlx_audio.tts.utils"] = MagicMock()
    sys.modules["mlx_audio.tts.utils"].load_model = mock_load_model
    sys.modules["mlx_audio.tts.generate"] = MagicMock()
    sys.modules["mlx_audio.tts.generate"].generate_audio = mock_generate_audio

    from text2speech.models import Text2SpeechRequest
    from text2speech.providers.fish_mlx import FishMLXProvider

    provider = FishMLXProvider()
    out_file = tmp_path / "out.wav"
    request = Text2SpeechRequest(text="Hello", reference_audio="/tmp/ref.wav")

    response = provider.generate(request, str(out_file))

    assert response.output_path == str(out_file)
    call_kwargs = mock_generate_audio.call_args[1]
    assert call_kwargs["ref_audio"] == "/tmp/ref.wav"
