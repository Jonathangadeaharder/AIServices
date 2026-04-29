import subprocess

import pytest
from aiservices_core.errors import ProviderError
from video2audio.models import Video2AudioRequest
from video2audio.providers.ffmpeg import FFmpegProvider


def test_ffmpeg_provider_init(mocker):
    mocker.patch("video2audio.providers.ffmpeg.shutil.which", return_value="/usr/bin/ffmpeg")
    provider = FFmpegProvider()
    assert provider.device == "auto"


def test_ffmpeg_not_found(mocker):
    mocker.patch("video2audio.providers.ffmpeg.shutil.which", return_value=None)
    provider = FFmpegProvider()
    with pytest.raises(ProviderError, match="ffmpeg not found"):
        provider._find_ffmpeg()


def test_ffmpeg_generate_wav(tmp_path, mocker):
    mocker.patch("video2audio.providers.ffmpeg.shutil.which", return_value="/usr/bin/ffmpeg")
    mock_run = mocker.patch("video2audio.providers.ffmpeg.subprocess.run")
    mock_run.return_value = mocker.MagicMock(stderr="")

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


def test_ffmpeg_generate_mp3(tmp_path, mocker):
    mocker.patch("video2audio.providers.ffmpeg.shutil.which", return_value="/usr/bin/ffmpeg")
    mock_run = mocker.patch("video2audio.providers.ffmpeg.subprocess.run")
    mock_run.return_value = mocker.MagicMock(stderr="")

    provider = FFmpegProvider()
    request = Video2AudioRequest(video_path="/tmp/test.mp4", output_format="mp3", mono=False)
    out = tmp_path / "out.mp3"
    response = provider.generate(request, str(out))

    assert response.metadata["codec"] == "libmp3lame"
    cmd = mock_run.call_args[0][0]
    assert "-ac" not in cmd


def test_ffmpeg_generate_default_output(tmp_path, mocker):
    mocker.patch("video2audio.providers.ffmpeg.shutil.which", return_value="/usr/bin/ffmpeg")
    mock_run = mocker.patch("video2audio.providers.ffmpeg.subprocess.run")
    mock_run.return_value = mocker.MagicMock(stderr="")

    provider = FFmpegProvider()
    request = Video2AudioRequest(video_path="/tmp/test.mp4", output_format="aac")
    response = provider.generate(request)

    assert response.output_path == "output.aac"
    assert response.metadata["codec"] == "aac"


def test_ffmpeg_generate_error(tmp_path, mocker):
    mocker.patch("video2audio.providers.ffmpeg.shutil.which", return_value="/usr/bin/ffmpeg")
    mock_run = mocker.patch("video2audio.providers.ffmpeg.subprocess.run")
    mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg", stderr="error")

    provider = FFmpegProvider()
    request = Video2AudioRequest(video_path="/tmp/test.mp4")
    with pytest.raises(ProviderError, match="FFmpeg failed"):
        provider.generate(request, str(tmp_path / "out.wav"))


def test_ffmpeg_timeout(tmp_path, mocker):
    mocker.patch("video2audio.providers.ffmpeg.shutil.which", return_value="/usr/bin/ffmpeg")
    mock_run = mocker.patch("video2audio.providers.ffmpeg.subprocess.run")
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=600)

    provider = FFmpegProvider()
    request = Video2AudioRequest(video_path="/tmp/test.mp4")
    with pytest.raises(ProviderError, match="timed out"):
        provider.generate(request, str(tmp_path / "out.wav"))


def test_ffmpeg_file_not_found(tmp_path, mocker):
    mocker.patch("video2audio.providers.ffmpeg.shutil.which", return_value="/usr/bin/ffmpeg")
    mock_run = mocker.patch("video2audio.providers.ffmpeg.subprocess.run")
    mock_run.side_effect = FileNotFoundError("no ffmpeg")

    provider = FFmpegProvider()
    request = Video2AudioRequest(video_path="/tmp/test.mp4")
    with pytest.raises(ProviderError, match="FFmpeg not found"):
        provider.generate(request, str(tmp_path / "out.wav"))
