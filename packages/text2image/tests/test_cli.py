from text2image.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_generate_success(tmp_path, mocker):
    mock_registry = mocker.patch("text2image.cli.registry")
    mock_provider = mocker.MagicMock()
    mock_response = mocker.MagicMock()
    mock_response.output_path = str(tmp_path / "out.png")
    mock_response.metadata = {"provider": "mlx"}
    mock_provider.generate.return_value = mock_response
    mock_registry.get.return_value = mock_provider

    out = tmp_path / "out.png"
    result = runner.invoke(
        app,
        ["--prompt", "a beautiful sunset", "--output", str(out)],
    )
    assert result.exit_code == 0


def test_generate_error(tmp_path, mocker):
    mock_registry = mocker.patch("text2image.cli.registry")
    mock_registry.get.side_effect = RuntimeError("failed")

    result = runner.invoke(
        app,
        ["--prompt", "test", "--output", "/tmp/out.png"],
    )
    assert result.exit_code == 1
