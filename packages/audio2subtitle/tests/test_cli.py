import re

from audio2subtitle.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_transcribe_success(tmp_path, mocker):
    mock_registry = mocker.patch("audio2subtitle.cli.registry")
    mock_provider = mocker.MagicMock()
    mock_response = mocker.MagicMock()
    mock_response.output_path = str(tmp_path / "out.srt")
    mock_response.entries = []
    mock_response.language = "en"
    mock_provider.generate.return_value = mock_response
    mock_registry.get.return_value = mock_provider

    out = tmp_path / "out.srt"
    result = runner.invoke(
        app,
        ["--input", "/tmp/audio.wav", "--output", str(out)],
    )
    assert result.exit_code == 0
    assert str(out) in result.output
    mock_registry.get.assert_called_once()
    mock_provider.generate.assert_called_once()


def test_transcribe_error(mocker):
    mock_registry = mocker.patch("audio2subtitle.cli.registry")
    mock_registry.get.side_effect = RuntimeError("provider failed")

    result = runner.invoke(
        app,
        ["--input", "/tmp/audio.wav", "--output", "/tmp/out.srt"],
    )
    assert result.exit_code == 1
    mock_registry.get.assert_called_once()


def test_transcribe_with_language(tmp_path, mocker):
    mock_registry = mocker.patch("audio2subtitle.cli.registry")
    mock_provider = mocker.MagicMock()
    mock_response = mocker.MagicMock()
    mock_response.output_path = str(tmp_path / "out.srt")
    mock_response.entries = []
    mock_response.language = "de"
    mock_provider.generate.return_value = mock_response
    mock_registry.get.return_value = mock_provider

    out = tmp_path / "out.srt"
    result = runner.invoke(
        app,
        ["--input", "/tmp/audio.wav", "--output", str(out), "--language", "de"],
    )
    assert result.exit_code == 0
    call_args = mock_provider.generate.call_args
    req = call_args[0][0]
    assert req.language == "de"


def test_transcribe_with_model(tmp_path, mocker):
    mock_registry = mocker.patch("audio2subtitle.cli.registry")
    mock_provider = mocker.MagicMock()
    mock_response = mocker.MagicMock()
    mock_response.output_path = str(tmp_path / "out.srt")
    mock_response.entries = []
    mock_response.language = "en"
    mock_provider.generate.return_value = mock_response
    mock_registry.get.return_value = mock_provider

    out = tmp_path / "out.srt"
    result = runner.invoke(
        app,
        ["--input", "/tmp/audio.wav", "--output", str(out), "--model", "custom-model"],
    )
    assert result.exit_code == 0
    call_args = mock_provider.generate.call_args
    req = call_args[0][0]
    assert req.model_name == "custom-model"


def test_help_shows_input_output(mocker):
    mocker.patch("audio2subtitle.cli.registry")
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    clean = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", result.output)
    assert "--input" in clean
    assert "--output" in clean
    assert "--language" in clean
    assert "--model" in clean
