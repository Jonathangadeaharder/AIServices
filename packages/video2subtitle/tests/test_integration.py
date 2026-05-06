import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from video2subtitle.cli import app

runner = CliRunner()


SRT_TIMESTAMP_RE = re.compile(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}")


@pytest.mark.integration
def test_transcribe_10s_video(test_video_10s, tmp_path, mlx_whisper_available):
    """Integration test: transcribe a 10s fixture mp4, assert SRT well-formed."""
    if not mlx_whisper_available:
        pytest.skip("mlx_whisper not available")
    output = tmp_path / "output.srt"

    result = runner.invoke(
        app,
        ["--input", str(test_video_10s), "--output", str(output)],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert output.exists(), "SRT file not created"

    content = output.read_text()
    lines = content.strip().split("\n")

    assert len(lines) >= 3, f"SRT too short: {len(lines)} lines"

    has_timestamp = any(SRT_TIMESTAMP_RE.match(line) for line in lines)
    assert has_timestamp, "No SRT timestamp found in output"

    assert "WEBVTT" not in content, "SRT file should not contain WEBVTT header"


@pytest.mark.integration
def test_transcribe_10s_video_default_output(test_video_10s, tmp_path, monkeypatch, mlx_whisper_available):
    """Integration test: default output path is <input>.srt."""
    if not mlx_whisper_available:
        pytest.skip("mlx_whisper not available")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        ["--input", str(test_video_10s)],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    expected_output = test_video_10s.with_suffix(".srt")
    assert expected_output.exists(), f"Default output not created at {expected_output}"


@pytest.mark.integration
def test_burn_in_produces_mp4(test_video_10s, tmp_path, mlx_whisper_available):
    """Integration test: --burn-in produces a new mp4 with hard-coded subtitles."""
    if not mlx_whisper_available:
        pytest.skip("mlx_whisper not available")
    output_srt = tmp_path / "output.srt"

    result = runner.invoke(
        app,
        ["--input", str(test_video_10s), "--output", str(output_srt), "--burn-in"],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"

    burned_video = test_video_10s.with_stem(test_video_10s.stem + "_subtitled")
    assert burned_video.exists(), f"Burned video not created at {burned_video}"

    import subprocess

    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_streams", "-print_format", "json", str(burned_video)],
        capture_output=True,
        text=True,
    )
    import json

    streams = json.loads(probe.stdout).get("streams", [])
    subtitle_streams = [s for s in streams if s.get("codec_type") == "subtitle"]
    assert len(subtitle_streams) == 0, f"Burned video has {len(subtitle_streams)} subtitle streams (should be 0)"
