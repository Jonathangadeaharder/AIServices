"""Integration test: CLI → provider → SRT output with a generated WAV fixture."""

import struct
import sys
import wave
from pathlib import Path

from typer.testing import CliRunner

runner = CliRunner()


def _generate_wav(path: Path, duration_s: float = 10.0, sample_rate: int = 16000):
    """Generate a minimal WAV file with silence."""
    n_samples = int(duration_s * sample_rate)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))


def test_cli_e2e_produces_srt(tmp_path, mocker):
    """Full CLI pipeline: WAV → mocked whisper → SRT with timestamps + text."""
    audio_file = tmp_path / "fixture.wav"
    _generate_wav(audio_file)

    srt_file = tmp_path / "out.srt"

    mock_whisper = mocker.MagicMock()
    mocker.patch.dict(sys.modules, {"mlx_whisper": mock_whisper})
    mock_whisper.transcribe.return_value = {
        "text": "Hello world. This is a test.",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "Hello world."},
            {"start": 2.5, "end": 5.0, "text": "This is a test."},
            {"start": 5.0, "end": 10.0, "text": "  "},
        ],
    }

    from audio2subtitle.cli import app

    result = runner.invoke(
        app,
        ["--input", str(audio_file), "--output", str(srt_file)],
    )

    assert result.exit_code == 0, result.output
    assert srt_file.exists()

    content = srt_file.read_text()
    lines = content.strip().split("\n")

    assert "-->" in content
    assert "Hello world." in content
    assert "This is a test." in content
    assert "00:00:00" in content
    assert "00:00:02" in content

    numbered_lines = [line for line in lines if line.strip().isdigit()]
    assert len(numbered_lines) >= 2


def test_cli_e2e_default_model_is_whisper_large_v3(tmp_path, mocker):
    """Verify default model is whisper-large-v3, not turbo."""
    audio_file = tmp_path / "fixture.wav"
    _generate_wav(audio_file, duration_s=1.0)

    srt_file = tmp_path / "out.srt"

    mock_whisper = mocker.MagicMock()
    mocker.patch.dict(sys.modules, {"mlx_whisper": mock_whisper})
    mock_whisper.transcribe.return_value = {
        "text": "Test.",
        "language": "en",
        "segments": [{"start": 0.0, "end": 1.0, "text": "Test."}],
    }

    from audio2subtitle.cli import app

    result = runner.invoke(
        app,
        ["--input", str(audio_file), "--output", str(srt_file)],
    )

    assert result.exit_code == 0, result.output
    call_args = mock_whisper.transcribe.call_args
    assert call_args[1]["path_or_hf_repo"] == "mlx-community/whisper-large-v3"
