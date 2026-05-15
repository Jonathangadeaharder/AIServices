from image2video.cli import app
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

def test_generate_success(mocker, tmp_path):
    mock_registry = mocker.patch("image2video.cli.registry")
    mock_provider = mocker.MagicMock()
    mock_response = mocker.MagicMock()
    mock_response.output_path = str(tmp_path / "out.mp4")
    mock_response.metadata = {"provider": "mlx"}
    mock_provider.generate.return_value = mock_response
    mock_registry.get.return_value = mock_provider

    out = tmp_path / "out.mp4"
    result = runner.invoke(
        app,
        ["--input", "/tmp/test.png", "--prompt", "test", "--output", str(out)],
    )
    assert result.exit_code == 0
    mock_registry.get.assert_called_once()
    mock_provider.generate.assert_called_once()

def test_generate_error(mocker):
    mock_registry = mocker.patch("image2video.cli.registry")
    mock_registry.get.side_effect = RuntimeError("failed")

    result = runner.invoke(
        app,
        ["--input", "/tmp/test.png", "--prompt", "test", "--output", "/tmp/out.mp4"],
    )
    assert result.exit_code == 1
    mock_registry.get.assert_called_once()

def test_generate_with_options(mocker, tmp_path):
    mock_registry = mocker.patch("image2video.cli.registry")
    mock_provider = _RecordingProvider()
    mock_registry.get.return_value = mock_provider

    out = tmp_path / "out.mp4"
    result = runner.invoke(
        app,
        ["--input", "/tmp/test.png", "--prompt", "test", "--output", str(out), "--width", "640", "--height", "640", "--seconds", "4", "--fps", "24", "--steps", "4"],
    )
    assert result.exit_code == 0
    assert mock_provider.last_request.num_frames == 96
    assert mock_provider.last_request.fps == 24
