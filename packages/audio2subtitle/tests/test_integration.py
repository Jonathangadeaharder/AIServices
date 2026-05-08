import sys

from audio2subtitle.models import Audio2SubtitleRequest
from audio2subtitle.providers.mlx import MLXWhisperProvider


def test_mlx_provider_init_default():
    provider = MLXWhisperProvider()
    assert provider is not None


def test_mlx_provider_srt_output(tmp_path, mocker):
    mock_whisper = mocker.MagicMock()
    mocker.patch.dict(sys.modules, {"mlx_whisper": mock_whisper})
    mock_whisper.transcribe.return_value = {
        "text": "Hello world.",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "Hello world."},
        ],
    }

    provider = MLXWhisperProvider()
    request = Audio2SubtitleRequest(audio_path="/tmp/audio.wav")
    out = tmp_path / "out.srt"
    response = provider.generate(request, str(out))

    assert response.output_path == str(out)
    assert len(response.entries) == 1
    assert response.entries[0].text == "Hello world."
    assert response.language == "en"

    content = (tmp_path / "out.srt").read_text()
    assert "-->" in content
    assert "Hello world." in content


def test_mlx_provider_vtt_output(tmp_path, mocker):
    mock_whisper = mocker.MagicMock()
    mocker.patch.dict(sys.modules, {"mlx_whisper": mock_whisper})
    mock_whisper.transcribe.return_value = {
        "text": "Hello.",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Hello."},
        ],
    }

    provider = MLXWhisperProvider()
    request = Audio2SubtitleRequest(audio_path="/tmp/audio.wav", output_format="vtt")
    out = tmp_path / "out.vtt"
    provider.generate(request, str(out))

    content = (tmp_path / "out.vtt").read_text()
    assert content.startswith("WEBVTT\n\n")


def test_mlx_provider_default_output(mocker):
    mock_whisper = mocker.MagicMock()
    mocker.patch.dict(sys.modules, {"mlx_whisper": mock_whisper})
    mock_whisper.transcribe.return_value = {
        "text": "Test.",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Test."},
        ],
    }

    provider = MLXWhisperProvider()
    request = Audio2SubtitleRequest(audio_path="/tmp/audio.wav")
    response = provider.generate(request, output_path=None)

    assert response.output_path == "output.srt"
