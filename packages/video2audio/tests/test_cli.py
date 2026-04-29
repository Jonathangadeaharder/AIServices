from typer.testing import CliRunner
from video2audio.cli import app

runner = CliRunner()


def test_extract_success(tmp_path, mocker):
    mock_registry = mocker.patch("video2audio.cli.registry")
    mock_provider = mocker.MagicMock()
    mock_response = mocker.MagicMock()
    mock_response.output_path = str(tmp_path / "out.wav")
    mock_response.duration_seconds = 10.5
    mock_response.metadata = {}
    mock_provider.generate.return_value = mock_response
    mock_registry.get.return_value = mock_provider

    out = tmp_path / "out.wav"
    result = runner.invoke(
        app,
        ["--video", "/tmp/video.mp4", "--output", str(out)],
    )
    assert result.exit_code == 0


def test_extract_error(mocker):
    mock_registry = mocker.patch("video2audio.cli.registry")
    mock_registry.get.side_effect = RuntimeError("failed")

    result = runner.invoke(
        app,
        ["--video", "/tmp/video.mp4", "--output", "/tmp/out.wav"],
    )
    assert result.exit_code == 1


def test_extract_stereo(tmp_path, mocker):
    mock_registry = mocker.patch("video2audio.cli.registry")
    mock_provider = mocker.MagicMock()
    mock_response = mocker.MagicMock()
    mock_response.output_path = str(tmp_path / "out.wav")
    mock_response.duration_seconds = None
    mock_response.metadata = {}
    mock_provider.generate.return_value = mock_response
    mock_registry.get.return_value = mock_provider

    out = tmp_path / "out.wav"
    result = runner.invoke(
        app,
        ["--video", "/tmp/video.mp4", "--output", str(out), "--stereo"],
    )
    assert result.exit_code == 0
