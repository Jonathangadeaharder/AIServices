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

    out = tmp_path / "out.srt"
    result = runner.invoke(
        app,
        ["--video", "/tmp/video.mp4", "--output", str(out)],
    )
    assert result.exit_code == 0
    assert str(out) in result.output


def test_transcribe_error(mocker):
    mock_registry = mocker.patch("video2subtitle.cli.registry")
    mock_registry.get.side_effect = RuntimeError("provider failed")

    result = runner.invoke(
        app,
        ["--video", "/tmp/video.mp4", "--output", "/tmp/out.srt"],
    )
    assert result.exit_code == 1


def test_transcribe_vtt_format(tmp_path, mocker):
    mock_registry = mocker.patch("video2subtitle.cli.registry")
    mock_provider = mocker.MagicMock()
    mock_response = mocker.MagicMock()
    mock_response.output_path = str(tmp_path / "out.vtt")
    mock_response.entries = []
    mock_response.language = None
    mock_provider.generate.return_value = mock_response
    mock_registry.get.return_value = mock_provider

    out = tmp_path / "out.vtt"
    result = runner.invoke(
        app,
        ["--video", "/tmp/video.mp4", "--output", str(out), "--format", "vtt"],
    )
    assert result.exit_code == 0
