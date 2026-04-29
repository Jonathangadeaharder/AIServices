from unittest.mock import MagicMock, patch

from image2video.cli import app
from typer.testing import CliRunner

runner = CliRunner()


@patch("image2video.cli.registry")
def test_generate_success(mock_registry, tmp_path):
    mock_provider = MagicMock()
    mock_response = MagicMock()
    mock_response.output_path = str(tmp_path / "out.mp4")
    mock_response.metadata = {"provider": "mlx"}
    mock_provider.generate.return_value = mock_response
    mock_registry.get.return_value = mock_provider

    out = tmp_path / "out.mp4"
    result = runner.invoke(
        app,
        [
            "--image", "/tmp/test.png",
            "--prompt", "a test video",
            "--output", str(out),
        ],
    )
    assert result.exit_code == 0


@patch("image2video.cli.registry")
def test_generate_error(mock_registry, tmp_path):
    mock_registry.get.side_effect = RuntimeError("failed")

    result = runner.invoke(
        app,
        ["--image", "/tmp/test.png", "--prompt", "test", "--output", "/tmp/out.mp4"],
    )
    assert result.exit_code == 1


@patch("image2video.cli.registry")
def test_generate_with_options(mock_registry, tmp_path):
    mock_provider = MagicMock()
    mock_response = MagicMock()
    mock_response.output_path = str(tmp_path / "out.mp4")
    mock_response.metadata = {"provider": "mlx"}
    mock_provider.generate.return_value = mock_response
    mock_registry.get.return_value = mock_provider

    out = tmp_path / "out.mp4"
    result = runner.invoke(
        app,
        [
            "--image", "/tmp/test.png",
            "--prompt", "test",
            "--output", str(out),
            "--width", "640",
            "--height", "640",
            "--frames", "81",
            "--steps", "4",
            "--fps", "16",
        ],
    )
    assert result.exit_code == 0
