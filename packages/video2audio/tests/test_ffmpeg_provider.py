import subprocess
from unittest.mock import MagicMock, patch

import pytest
from aiservices_core.errors import ProviderError
from video2audio.models import Video2AudioRequest
from video2audio.providers.ffmpeg import FFmpegProvider


@patch("video2audio.providers.ffmpeg.shutil.which", return_value="/usr/bin/ffmpeg")
def test_ffmpeg_provider_init(mock_which):
    provider = FFmpegProvider()
    assert provider.device == "auto"


@patch("video2audio.providers.ffmpeg.shutil.which", return_value=None)
def test_ffmpeg_not_found(mock_which):
    provider = FFmpegProvider()
    with pytest.raises(ProviderError, match="ffmpeg not found"):
        provider._find_ffmpeg()


@patch("video2audio.providers.ffmpeg.subprocess.run")
@patch("video2audio.providers.ffmpeg.shutil.which", return_value="/usr/bin/ffmpeg")
def test_ffmpeg_generate_wav(mock_which, mock_run, tmp_path):
    mock_run.return_value = MagicMock(stderr="")

    provider = FFmpegProvider()
    request = Video2AudioRequest(video_path="/tmp/test.mp4")
    out = tmp_path / "out.wav"
    response = provider.generate(request, str(out))

    assert response.output_path == str(out)
    assert response.metadata["provider"] == "ffmpeg"
    assert response.metadata["codec"] == "pcm_s16le"
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "-ac" in cmd


@patch("video2audio.providers.ffmpeg.subprocess.run")
@patch("video2audio.providers.ffmpeg.shutil.which", return_value="/usr/bin/ffmpeg")
def test_ffmpeg_generate_mp3(mock_which, mock_run, tmp_path):
    mock_run.return_value = MagicMock(stderr="")

    provider = FFmpegProvider()
    request = Video2AudioRequest(video_path="/tmp/test.mp4", output_format="mp3", mono=False)
    out = tmp_path / "out.mp3"
    response = provider.generate(request, str(out))

    assert response.metadata["codec"] == "libmp3lame"
    cmd = mock_run.call_args[0][0]
    assert "-ac" not in cmd


@patch("video2audio.providers.ffmpeg.subprocess.run")
@patch("video2audio.providers.ffmpeg.shutil.which", return_value="/usr/bin/ffmpeg")
def test_ffmpeg_generate_default_output(mock_which, mock_run, tmp_path):
    mock_run.return_value = MagicMock(stderr="")

    provider = FFmpegProvider()
    request = Video2AudioRequest(video_path="/tmp/test.mp4", output_format="aac")
    response = provider.generate(request)

    assert response.output_path == "output.aac"
    assert response.metadata["codec"] == "aac"


@patch("video2audio.providers.ffmpeg.subprocess.run")
@patch("video2audio.providers.ffmpeg.shutil.which", return_value="/usr/bin/ffmpeg")
def test_ffmpeg_generate_error(mock_which, mock_run, tmp_path):
    mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg", stderr="error")

    provider = FFmpegProvider()
    request = Video2AudioRequest(video_path="/tmp/test.mp4")
    with pytest.raises(ProviderError, match="FFmpeg failed"):
        provider.generate(request, str(tmp_path / "out.wav"))


@patch("video2audio.providers.ffmpeg.subprocess.run")
@patch("video2audio.providers.ffmpeg.shutil.which", return_value="/usr/bin/ffmpeg")
def test_ffmpeg_timeout(mock_which, mock_run, tmp_path):
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=600)

    provider = FFmpegProvider()
    request = Video2AudioRequest(video_path="/tmp/test.mp4")
    with pytest.raises(ProviderError, match="timed out"):
        provider.generate(request, str(tmp_path / "out.wav"))


@patch("video2audio.providers.ffmpeg.subprocess.run")
@patch("video2audio.providers.ffmpeg.shutil.which", return_value="/usr/bin/ffmpeg")
def test_ffmpeg_file_not_found(mock_which, mock_run, tmp_path):
    mock_run.side_effect = FileNotFoundError("no ffmpeg")

    provider = FFmpegProvider()
    request = Video2AudioRequest(video_path="/tmp/test.mp4")
    with pytest.raises(ProviderError, match="FFmpeg not found"):
        provider.generate(request, str(tmp_path / "out.wav"))
