from audio2subtitle.cli import app
from typer.testing import CliRunner

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
    mock_registry = mocker.patch("audio2subtitle.cli.registry")
    mock_provider = mocker.MagicMock()
    mock_response = mocker.MagicMock()
    mock_response.output_path = str(tmp_path / "out.srt")
    mock_provider.generate.return_value = mock_response
    mock_registry.get.return_value = mock_provider

    out = tmp_path / "out.srt"
    result = runner.invoke(
        app,
        ["--input", "/tmp/audio.wav", "--output", str(out)],
    )
    assert result.exit_code == 0
    mock_registry.get.assert_called_once()
    mock_provider.generate.assert_called_once()


def test_transcribe_error(mocker):
    mock_registry = mocker.patch("audio2subtitle.cli.registry")
    mock_registry.get.side_effect = RuntimeError("failed")

    result = runner.invoke(
        app,
        ["--input", "/tmp/audio.wav", "--output", "/tmp/out.srt"],
    )
    assert result.exit_code == 1
    mock_registry.get.assert_called_once()


def test_transcribe_with_language(tmp_path, mocker):
    mock_registry = mocker.patch("audio2subtitle.cli.registry")
    mock_provider = _RecordingProvider()
    mock_registry.get.return_value = mock_provider

    out = tmp_path / "out.srt"
    result = runner.invoke(
        app,
        ["--input", "/tmp/audio.wav", "--output", str(out), "--language", "de"],
    )
    assert result.exit_code == 0
    assert mock_provider.last_request.language == "de"


def test_transcribe_with_model(tmp_path, mocker):
    mock_registry = mocker.patch("audio2subtitle.cli.registry")
    mock_provider = _RecordingProvider()
    mock_registry.get.return_value = mock_provider

    out = tmp_path / "out.srt"
    result = runner.invoke(
        app,
        ["--input", "/tmp/audio.wav", "--output", str(out), "--model", "custom-model"],
    )
    assert result.exit_code == 0
    assert mock_provider.last_request.model_name == "custom-model"


def test_help_shows_input_output(mocker):
    mocker.patch("audio2subtitle.cli.registry")
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    clean = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", result.output)
    assert "--input" in clean
    assert "--output" in clean
    assert "--language" in clean
