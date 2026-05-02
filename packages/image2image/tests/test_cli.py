from image2image.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_generate_success(mocker, tmp_path):
    mock_registry = mocker.patch("image2image.cli.registry")
    mock_provider = mocker.MagicMock()
    mock_response = mocker.MagicMock()
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
    mock_registry.get.assert_called_once()
    mock_provider.generate.assert_called_once()


def test_generate_error(mocker, tmp_path):
    mock_registry = mocker.patch("image2image.cli.registry")
    mock_registry.get.side_effect = RuntimeError("failed")

    result = runner.invoke(
        app,
        ["--input", "/tmp/test.png", "--prompt", "test", "--output", "/tmp/out.png"],
    )
    assert result.exit_code == 1
    mock_registry.get.assert_called_once()


def test_generate_custom_strength(mocker, tmp_path):
    mock_registry = mocker.patch("image2image.cli.registry")
    mock_provider = mocker.MagicMock()
    mock_response = mocker.MagicMock()
    mock_response.output_path = str(tmp_path / "out.png")
    mock_response.metadata = {"provider": "mlx"}
    mock_provider.generate.return_value = mock_response
    mock_registry.get.return_value = mock_provider

    dummy_input = tmp_path / "input.png"
    dummy_input.write_bytes(b"fake")

    out = tmp_path / "out.png"
    result = runner.invoke(
        app,
        [
            "--input", str(dummy_input),
            "--prompt", "test",
            "--output", str(out),
            "--strength", "0.8",
        ],
    )
    assert result.exit_code == 0
    call_args = mock_provider.generate.call_args
    req = call_args[0][0]
    assert req.strength == 0.8
