from typer.testing import CliRunner
from video2subtitle.cli import app

runner = CliRunner()

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
    mock_provider = _RecordingProvider()
    mock_registry.get.return_value = mock_provider

    video = tmp_path / "video.mp4"
    video.touch()
    result = runner.invoke(
        app,
        ["--input", str(video)],
    )
    assert result.exit_code == 0
    assert mock_provider.last_output_path == str(tmp_path / "video.srt")

def test_transcribe_language(tmp_path, mocker):
    mock_registry = mocker.patch("video2subtitle.cli.registry")
    mock_provider = _RecordingProvider()
    mock_registry.get.return_value = mock_provider

    video = tmp_path / "video.mp4"
    video.touch()
    result = runner.invoke(
        app,
        ["--input", str(video), "--output", str(tmp_path / "out.srt"), "--language", "en"],
    )
    assert result.exit_code == 0
    assert mock_provider.last_request.language == "en"

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
