import shutil
import subprocess

import pytest


@pytest.fixture(scope="session")
def ffmpeg_available():
    """Check if ffmpeg is available."""
    return shutil.which("ffmpeg") is not None


@pytest.fixture(scope="session")
def ffmpeg_subtitles_filter_available(ffmpeg_available):
    """Check if ffmpeg has the subtitles filter (requires libass)."""
    if not ffmpeg_available:
        return False
    result = subprocess.run(
        ["ffmpeg", "-filters"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return "subtitles" in result.stdout


@pytest.fixture(scope="session")
def mlx_whisper_available():
    """Check if mlx_whisper is available."""
    try:
        import mlx_whisper  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def test_video_10s(tmp_path_factory, ffmpeg_available):
    """Create a 10-second test video with audio."""
    if not ffmpeg_available:
        pytest.skip("ffmpeg not available")

    tmp_dir = tmp_path_factory.mktemp("videos")
    video_path = tmp_dir / "test_10s.mp4"

    cmd = [
        "ffmpeg",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=10",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=320x240:d=10",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-shortest",
        "-y",
        str(video_path),
    ]

    subprocess.run(cmd, check=True, capture_output=True, timeout=30)
    return video_path
