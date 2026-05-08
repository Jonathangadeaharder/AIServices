from video2audio.models import Video2AudioRequest
from video2audio.providers.ffmpeg import FFmpegProvider


def test_ffmpeg_provider_init_default():
    provider = FFmpegProvider()
    assert provider.device == "auto"


def test_ffmpeg_provider_extract_wav(mocker, tmp_path):
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
    assert "-vn" in cmd


def test_ffmpeg_provider_extract_mp3(mocker, tmp_path):
    mocker.patch("video2audio.providers.ffmpeg.shutil.which", return_value="/usr/bin/ffmpeg")
    mock_run = mocker.patch("video2audio.providers.ffmpeg.subprocess.run")
    mock_run.return_value = mocker.MagicMock(stderr="")

    provider = FFmpegProvider()
    request = Video2AudioRequest(video_path="/tmp/test.mp4", output_format="mp3", mono=False)
    out = tmp_path / "out.mp3"
    response = provider.generate(request, str(out))

    assert response.metadata["codec"] == "libmp3lame"
    assert response.metadata["mono"] is False


def test_ffmpeg_provider_default_output(mocker):
    mocker.patch("video2audio.providers.ffmpeg.shutil.which", return_value="/usr/bin/ffmpeg")
    mock_run = mocker.patch("video2audio.providers.ffmpeg.subprocess.run")
    mock_run.return_value = mocker.MagicMock(stderr="")

    provider = FFmpegProvider()
    request = Video2AudioRequest(video_path="/tmp/test.mp4", output_format="aac")
    response = provider.generate(request, output_path=None)

    assert response.output_path == "output.aac"
