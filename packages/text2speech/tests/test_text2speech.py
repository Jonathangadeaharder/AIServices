import importlib.util
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

_HAS_ORMSGPACK = importlib.util.find_spec("ormsgpack") is not None

# Module keys that get inject-mocked for mlx_audio tests
_MLX_MODULES = ["mlx_audio", "mlx_audio.tts", "mlx_audio.tts.utils", "mlx_audio.tts.generate"]


@pytest.fixture(autouse=True)
def _cleanup_sys_modules():
    """Save and restore sys.modules entries that tests mutate."""
    saved = {}
    for key in list(_MLX_MODULES):
        saved[key] = sys.modules.get(key)
    saved["ormsgpack"] = sys.modules.get("ormsgpack")
    yield
    for key in list(_MLX_MODULES):
        if saved.get(key) is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = saved[key]
    if saved.get("ormsgpack") is None:
        sys.modules.pop("ormsgpack", None)
    else:
        sys.modules["ormsgpack"] = saved["ormsgpack"]


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

    assert response.output_path == str(out_file), "output_path should match"
    assert response.metadata["provider"] == "fish-speech-api", "provider should be fish-speech-api"
    assert out_file.exists(), "output file should exist"
    assert out_file.read_bytes() == b"fake_audio_data", "output should contain audio data"


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

    assert response.output_path == str(out_file), "output_path should match"
    assert response.metadata["provider"] == "fish-speech-api", "provider should be fish-speech-api"
    assert out_file.exists(), "output file should exist"


def test_fish_provider_build_text():
    from text2speech.models import Text2SpeechRequest
    from text2speech.providers.fish import FishSpeechProvider

    provider = FishSpeechProvider()

    req = Text2SpeechRequest(text="Hello")
    assert provider._build_fish_text(req) == "Hello", "plain text"

    req_emotion = Text2SpeechRequest(text="Hello", emotion="happy")
    assert provider._build_fish_text(req_emotion) == "[happy] Hello", "emotion prefix"

    req_tone = Text2SpeechRequest(text="Hello", tone="calm")
    assert provider._build_fish_text(req_tone) == "[calm] Hello", "tone prefix"

    req_effect = Text2SpeechRequest(text="Hello", effect="echo")
    assert provider._build_fish_text(req_effect) == "[echo] Hello", "effect prefix"


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
    assert provider._build_text(req) == "Hello", "plain text"

    req_emotion = Text2SpeechRequest(text="Hello", emotion="happy")
    assert provider._build_text(req_emotion) == "[happy] Hello", "emotion prefix"

    req_tone = Text2SpeechRequest(text="Hello", tone="calm")
    assert provider._build_text(req_tone) == "[calm] Hello", "tone prefix"

    req_effect = Text2SpeechRequest(text="Hello", effect="echo")
    assert provider._build_text(req_effect) == "[echo] Hello", "effect prefix"

    req_all = Text2SpeechRequest(text="Hello", emotion="happy", tone="calm", effect="echo")
    assert provider._build_text(req_all) == "[happy][calm][echo] Hello", "all modifiers"


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
