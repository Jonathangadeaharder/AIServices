from pathlib import Path

from typer.testing import CliRunner
from video2subtitle.cli import app

runner = CliRunner()


def test_transcribe_success(tmp_path, mocker):
    mock_registry = mocker.patch("video2subtitle.cli.registry")
    mock_provider = mocker.MagicMock()
    mock_response = mocker.MagicMock()
    mock_response.output_path = str(tmp_path / "out.srt")
    mock_response.entries = []
    mock_response.language = "en"
    mock_provider.generate.return_value = mock_response
    mock_registry.get.return_value = mock_provider

    video = tmp_path / "video.mp4"
    video.touch()
    out = tmp_path / "out.srt"
    result = runner.invoke(
        app,
        ["--input", str(video), "--output", str(out)],
    )
    assert result.exit_code == 0
    assert str(out) in result.output
    mock_registry.get.assert_called_once()
    mock_provider.generate.assert_called_once()


def test_transcribe_error(mocker):
    mock_registry = mocker.patch("video2subtitle.cli.registry")
    mock_registry.get.side_effect = RuntimeError("provider failed")

    result = runner.invoke(
        app,
        ["--input", "/tmp/nonexistent.mp4", "--output", "/tmp/out.srt"],
    )
    assert result.exit_code == 1


def test_transcribe_default_output(tmp_path, mocker):
    mock_registry = mocker.patch("video2subtitle.cli.registry")
    mock_provider = mocker.MagicMock()
    mock_response = mocker.MagicMock()
    mock_response.output_path = str(tmp_path / "video.srt")
    mock_response.entries = []
    mock_response.language = None
    mock_provider.generate.return_value = mock_response
    mock_registry.get.return_value = mock_provider

    video = tmp_path / "video.mp4"
    video.touch()
    result = runner.invoke(
        app,
        ["--input", str(video)],
    )
    assert result.exit_code == 0
    call_args = mock_provider.generate.call_args
    assert call_args[0][1] == str(tmp_path / "video.srt")


def test_transcribe_language(tmp_path, mocker):
    mock_registry = mocker.patch("video2subtitle.cli.registry")
    mock_provider = mocker.MagicMock()
    mock_response = mocker.MagicMock()
    mock_response.output_path = str(tmp_path / "out.srt")
    mock_response.entries = []
    mock_response.language = "en"
    mock_provider.generate.return_value = mock_response
    mock_registry.get.return_value = mock_provider

    video = tmp_path / "video.mp4"
    video.touch()
    result = runner.invoke(
        app,
        ["--input", str(video), "--output", str(tmp_path / "out.srt"), "--language", "en"],
    )
    assert result.exit_code == 0
    call_args = mock_provider.generate.call_args
    req = call_args[0][0]
    assert req.language == "en"


def test_burn_in_flag(tmp_path, mocker):
    mock_registry = mocker.patch("video2subtitle.cli.registry")
    mock_provider = mocker.MagicMock()
    mock_response = mocker.MagicMock()
    mock_response.output_path = str(tmp_path / "out.srt")
    mock_response.entries = []
    mock_response.language = "en"
    mock_provider.generate.return_value = mock_response
    mock_registry.get.return_value = mock_provider

    mocker.patch("video2subtitle.cli._burn_subtitles")

    video = tmp_path / "video.mp4"
    video.touch()
    result = runner.invoke(
        app,
        ["--input", str(video), "--output", str(tmp_path / "out.srt"), "--burn-in"],
    )
    assert result.exit_code == 0
    assert "Burned-in video saved to" in result.output


def test_input_not_found(mocker):
    mocker.patch("video2subtitle.cli.registry")
    result = runner.invoke(
        app,
        ["--input", "/tmp/nonexistent_video.mp4"],
    )
    assert result.exit_code == 1
    assert "Input file not found" in result.output
