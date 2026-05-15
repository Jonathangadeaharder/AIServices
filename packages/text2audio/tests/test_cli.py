from text2audio.cli import app
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
        self.entries = []
        self.language = None

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
    mock_registry.get.assert_called_once()
    mock_provider.generate.assert_called_once()

def test_generate_error(mocker):
    mock_registry = mocker.patch("text2audio.cli.registry")
    mock_registry.get.side_effect = RuntimeError("failed")

    result = runner.invoke(
        app,
        ["--prompt", "test", "--output", "/tmp/out.wav"],
    )
    assert result.exit_code == 1
    mock_registry.get.assert_called_once()

def test_generate_with_seed(tmp_path, mocker):
    mock_registry = mocker.patch("text2audio.cli.registry")
    mock_provider = _RecordingProvider()
    mock_registry.get.return_value = mock_provider

    out = tmp_path / "out.wav"
    result = runner.invoke(
        app,
        ["--prompt", "test", "--output", str(out), "--seed", "42"],
    )
    assert result.exit_code == 0
    assert mock_provider.last_request.seed == 42

def test_generate_with_category_and_duration(tmp_path, mocker):
    mock_registry = mocker.patch("text2audio.cli.registry")
    mock_provider = _RecordingProvider()
    mock_registry.get.return_value = mock_provider

    out = tmp_path / "out.wav"
    result = runner.invoke(
        app,
        ["--prompt", "test", "--output", str(out), "--category", "ambient", "--duration", "30.0"],
    )
    assert result.exit_code == 0
    assert mock_provider.last_request.category.value == "ambient"
    assert mock_provider.last_request.duration_seconds == 30.0
