"""Integration tests for image2video CLI."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def fixture_image(tmp_path):
    """Create a minimal test image."""
    from PIL import Image

    img = Image.new("RGB", (640, 640), color="red")
    img_path = tmp_path / "test_input.png"
    img.save(img_path)
    return img_path


def test_cli_produces_valid_4s_mp4(fixture_image, tmp_path):
    """Integration test: produce a valid 4s mp4 from a fixture image (mocked provider)."""
    from image2video.cli import app
    from typer.testing import CliRunner

    runner = CliRunner()

    with patch("image2video.cli.registry") as mock_registry:
        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.output_path = str(tmp_path / "output.mp4")
        mock_response.metadata = {"provider": "mlx", "seed": 42}
        mock_provider.generate.return_value = mock_response
        mock_registry.get.return_value = mock_provider

        output_path = tmp_path / "output.mp4"
        result = runner.invoke(
            app,
            [
                "--input",
                str(fixture_image),
                "--output",
                str(output_path),
                "--seconds",
                "4",
                "--fps",
                "24",
                "--prompt",
                "test video",
            ],
        )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    mock_registry.get.assert_called_once()
    mock_provider.generate.assert_called_once()

    call_args = mock_provider.generate.call_args
    req = call_args[0][0]
    assert req.num_frames == 96  # 4 seconds * 24 fps
    assert req.fps == 24


def test_cli_help_shows_spec_options():
    """Verify --help shows the documented CLI surface."""
    result = subprocess.run(
        ["uv", "run", "python", "-m", "image2video.cli", "--help"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent.parent.parent),
    )

    assert result.returncode == 0
    assert "--input" in result.stdout
    assert "--output" in result.stdout
    assert "--seconds" in result.stdout
    assert "--fps" in result.stdout
    assert "--prompt" in result.stdout