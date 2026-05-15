import re

from typer.testing import CliRunner
from video2audio.cli import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

class _RecordingProvider:
    """Test double that captures requests instead of mocking call patterns."""
    def __init__(self):
        self.last_request = None
        self.last_output_path = None

    def generate(self, request, output_path):
        self.last_request = request
        self.last_output_path = output_path
        return _ProviderResponse(output_path)


class _ProviderResponse:
    """Minimal response stub for provider.generate()."""
    def __init__(self, output_path):
        self.output_path = output_path
        self.metadata = {}
        self.duration_seconds = None
        self.entries = []
        self.language = None

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
        ["--input", "/tmp/video.mp4", "--output", str(out)],
    )
    assert result.exit_code == 0
    mock_registry.get.assert_called_once()
    mock_provider.generate.assert_called_once()

def test_extract_error(mocker):
    mock_registry = mocker.patch("video2audio.cli.registry")
    mock_registry.get.side_effect = RuntimeError("failed")

    result = runner.invoke(
        app,
        ["--input", "/tmp/video.mp4", "--output", "/tmp/out.wav"],
    )
    assert result.exit_code == 1
    mock_registry.get.assert_called_once()

def test_extract_codec_option(tmp_path, mocker):
    mock_registry = mocker.patch("video2audio.cli.registry")
    mock_provider = _RecordingProvider()
    mock_registry.get.return_value = mock_provider

    out = tmp_path / "out.mp3"
    result = runner.invoke(
        app,
        ["--input", "/tmp/video.mp4", "--output", str(out), "--codec", "mp3"],
    )
    assert result.exit_code == 0
    assert mock_provider.last_request.output_format == "mp3"

def test_help_shows_options():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    clean = _ANSI_RE.sub("", result.output)
    assert "--input" in clean
    assert "--output" in clean
    assert "--codec" in clean
