import sys

from video2subtitle.models import Video2SubtitleRequest
from video2subtitle.providers.mlx import MLXProvider


def test_mlx_provider_init_default():
    provider = MLXProvider()
    assert provider is not None


def test_mlx_provider_srt_output(tmp_path, mocker):
    mock_whisper = mocker.MagicMock()
    mocker.patch.dict(sys.modules, {"mlx_whisper": mock_whisper})
    mocker.patch("video2subtitle.providers.mlx.shutil.which", return_value="/usr/bin/ffmpeg")
    mocker.patch("video2subtitle.providers.mlx.subprocess.run")

    mock_whisper.transcribe.return_value = {
        "text": "Hello world",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "Hello world"},
        ],
    }

    provider = MLXProvider()
    request = Video2SubtitleRequest(video_path="/tmp/video.mp4")
    output = str(tmp_path / "out.srt")

    response = provider.generate(request, output)

    assert response.output_path == output
    assert len(response.entries) == 1
    assert response.language == "en"

    content = (tmp_path / "out.srt").read_text()
    assert "-->" in content
    assert "Hello world" in content


def test_mlx_provider_vtt_output(tmp_path, mocker):
    mock_whisper = mocker.MagicMock()
    mocker.patch.dict(sys.modules, {"mlx_whisper": mock_whisper})
    mocker.patch("video2subtitle.providers.mlx.shutil.which", return_value="/usr/bin/ffmpeg")
    mocker.patch("video2subtitle.providers.mlx.subprocess.run")

    mock_whisper.transcribe.return_value = {
        "text": "Test",
        "language": "en",
        "segments": [
            {"start": 1.0, "end": 3.0, "text": "Test"},
        ],
    }

    provider = MLXProvider()
    request = Video2SubtitleRequest(video_path="/tmp/video.mp4", output_format="vtt")
    output = str(tmp_path / "out.vtt")

    provider.generate(request, output)

    content = (tmp_path / "out.vtt").read_text()
    assert content.startswith("WEBVTT")
    assert "-->" in content


def test_mlx_provider_default_output(mocker, tmp_path, monkeypatch):
    mock_whisper = mocker.MagicMock()
    mocker.patch.dict(sys.modules, {"mlx_whisper": mock_whisper})
    mocker.patch("video2subtitle.providers.mlx.shutil.which", return_value="/usr/bin/ffmpeg")
    mocker.patch("video2subtitle.providers.mlx.subprocess.run")

    mock_whisper.transcribe.return_value = {
        "text": "Test",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Test"},
        ],
    }

    monkeypatch.chdir(tmp_path)
    provider = MLXProvider()
    request = Video2SubtitleRequest(video_path="/tmp/video.mp4", output_format="vtt")
    response = provider.generate(request, output_path=None)

    assert response.output_path == "output.vtt"
    assert (tmp_path / "output.vtt").exists()
