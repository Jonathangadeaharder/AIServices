import json
import subprocess

import pytest
from typer.testing import CliRunner
from video2audio.cli import app

runner = CliRunner()


@pytest.fixture
def fixture_video(tmp_path):
    """Create a 2-second test video with audio using ffmpeg."""
    video_path = tmp_path / "test_input.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:d=2",
            "-shortest",
            str(video_path),
        ],
        capture_output=True,
        check=True,
        timeout=30,
    )
    return video_path


def _ffprobe_streams(path: str) -> list[dict]:
    """Return list of stream dicts from ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return json.loads(result.stdout)["streams"]


def _ffprobe_duration(path: str) -> float:
    """Return duration in seconds from ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def test_integration_extract_wav(fixture_video, tmp_path):
    """Extract audio from fixture mp4, assert ffprobe reports audio stream and duration."""
    output = tmp_path / "output.wav"
    result = runner.invoke(
        app,
        ["--input", str(fixture_video), "--output", str(output), "--codec", "wav"],
    )
    assert result.exit_code == 0
    assert output.exists()

    streams = _ffprobe_streams(str(output))
    audio_streams = [s for s in streams if s["codec_type"] == "audio"]
    assert len(audio_streams) == 1
    assert audio_streams[0]["codec_name"] == "pcm_s16le"

    duration = _ffprobe_duration(str(output))
    assert 1.5 <= duration <= 3.0, f"Expected ~2s duration, got {duration:.2f}s"


def test_integration_extract_mp3(fixture_video, tmp_path):
    """Extract audio as mp3, assert ffprobe reports mp3 stream."""
    output = tmp_path / "output.mp3"
    result = runner.invoke(
        app,
        ["--input", str(fixture_video), "--output", str(output), "--codec", "mp3"],
    )
    assert result.exit_code == 0
    assert output.exists()

    streams = _ffprobe_streams(str(output))
    audio_streams = [s for s in streams if s["codec_type"] == "audio"]
    assert len(audio_streams) == 1
    assert audio_streams[0]["codec_name"] == "mp3"
