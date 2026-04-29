from video2audio.models import Video2AudioRequest


def test_model_defaults():
    """Test that default values are sensible."""
    req = Video2AudioRequest(video_path="/tmp/test.mp4")
    assert req.output_format == "wav"
    assert req.sample_rate == 44100
    assert req.mono is True


def test_model_valid_formats():
    """Test that valid formats are accepted."""
    for fmt in ("wav", "mp3", "aac"):
        req = Video2AudioRequest(video_path="/tmp/test.mp4", output_format=fmt)
        assert req.output_format == fmt


def test_ffmpeg_provider_generate(tmp_path, mocker):
    from video2audio.providers.ffmpeg import FFmpegProvider

    provider = FFmpegProvider()
    out_file = tmp_path / "out.wav"

    request = Video2AudioRequest(video_path="/tmp/test.mp4")
    mocker.patch("video2audio.providers.ffmpeg.shutil.which", return_value="/usr/bin/ffmpeg")
    mock_run = mocker.patch("video2audio.providers.ffmpeg.subprocess.run")

    response = provider.generate(request, str(out_file))

    assert response.output_path == str(out_file)
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "-vn" in cmd
