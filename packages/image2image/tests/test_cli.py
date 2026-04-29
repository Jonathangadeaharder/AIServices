from unittest.mock import MagicMock, patch

from image2image.cli import app
from typer.testing import CliRunner

runner = CliRunner()


@patch("image2image.cli.registry")
def test_generate_success(mock_registry, tmp_path):
    mock_provider = MagicMock()
    mock_response = MagicMock()
    mock_response.output_path = str(tmp_path / "out.png")
    mock_response.metadata = {"provider": "mlx"}
    mock_provider.generate.return_value = mock_response
    mock_registry.get.return_value = mock_provider

    dummy_input = tmp_path / "input.png"
    dummy_input.write_bytes(b"fake")

    out = tmp_path / "out.png"
    result = runner.invoke(
        app,
        ["--input", str(dummy_input), "--prompt", "a test image", "--output", str(out)],
    )
    assert result.exit_code == 0


@patch("image2image.cli.registry")
def test_generate_error(mock_registry, tmp_path):
    mock_registry.get.side_effect = RuntimeError("failed")

    result = runner.invoke(
        app,
        ["--input", "/tmp/test.png", "--prompt", "test", "--output", "/tmp/out.png"],
    )
    assert result.exit_code == 1
