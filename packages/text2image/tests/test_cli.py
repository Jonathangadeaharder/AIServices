from text2image.cli import app
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
    mock_registry.get.assert_called_once()
    mock_provider.generate.assert_called_once()

def test_generate_error(tmp_path, mocker):
    mock_registry = mocker.patch("text2image.cli.registry")
    mock_registry.get.side_effect = RuntimeError("failed")

    result = runner.invoke(
        app,
        ["--prompt", "test", "--output", "/tmp/out.png"],
    )
    assert result.exit_code == 1
    mock_registry.get.assert_called_once()

def test_generate_custom_dimensions(tmp_path, mocker):
    mock_registry = mocker.patch("text2image.cli.registry")
    mock_provider = _RecordingProvider()
    mock_registry.get.return_value = mock_provider

    out = tmp_path / "out.png"
    result = runner.invoke(
        app,
        [
            "--prompt",
            "test",
            "--output",
            str(out),
            "--width",
            "512",
            "--height",
            "512",
            "--seed",
            "42",
        ],
    )
    assert result.exit_code == 0, "CLI should exit successfully"
    assert mock_provider.last_request.width == 512, "width should be 512"
    assert mock_provider.last_request.height == 512, "height should be 512"
    assert mock_provider.last_request.seed == 42, "seed should be 42"
