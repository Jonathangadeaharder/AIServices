import os

import pytest
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


@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS") != "1",
    reason="Requires RUN_INTEGRATION_TESTS=1",
)
def test_ffmpeg_provider_integration(tmp_path):
    """Integration test requiring FFmpeg and a valid video file."""
    from video2audio.providers.ffmpeg import FFmpegProvider

    provider = FFmpegProvider()
    out_file = tmp_path / "out.wav"

    request = Video2AudioRequest(
        video_path=os.environ["TEST_VIDEO_PATH"],
    )
    response = provider.generate(request, str(out_file))
    assert os.path.exists(response.output_path)
