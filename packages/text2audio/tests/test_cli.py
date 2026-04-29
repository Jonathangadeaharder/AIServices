from text2audio.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_generate_success(tmp_path, mocker):
    mock_registry = mocker.patch("text2audio.cli.registry")
    mock_provider = mocker.MagicMock()
    mock_response = mocker.MagicMock()
    mock_response.output_path = str(tmp_path / "out.wav")
    mock_provider.generate.return_value = mock_response
    mock_registry.get.return_value = mock_provider

    out = tmp_path / "out.wav"
    result = runner.invoke(
        app,
        ["--prompt", "calm piano music", "--output", str(out)],
    )
    assert result.exit_code == 0


def test_generate_error(mocker):
    mock_registry = mocker.patch("text2audio.cli.registry")
    mock_registry.get.side_effect = RuntimeError("failed")

    result = runner.invoke(
        app,
        ["--prompt", "test", "--output", "/tmp/out.wav"],
    )
    assert result.exit_code == 1


def test_generate_with_seed(tmp_path, mocker):
    mock_registry = mocker.patch("text2audio.cli.registry")
    mock_provider = mocker.MagicMock()
    mock_response = mocker.MagicMock()
    mock_response.output_path = str(tmp_path / "out.wav")
    mock_provider.generate.return_value = mock_response
    mock_registry.get.return_value = mock_provider

    out = tmp_path / "out.wav"
    result = runner.invoke(
        app,
        ["--prompt", "test", "--output", str(out), "--seed", "42"],
    )
    assert result.exit_code == 0
