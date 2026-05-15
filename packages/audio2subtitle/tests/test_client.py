from unittest.mock import MagicMock, patch

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

class TestGenerate:
    @patch("audio2subtitle.providers.registry")
    def test_calls_provider(self, mock_registry):
        from audio2subtitle.client import generate

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.output_path = "/tmp/out.srt"
        mock_provider.generate.return_value = mock_response
        mock_registry.get.return_value = mock_provider

        result = generate("/tmp/in.wav", "/tmp/out.srt")

        mock_registry.get.assert_called_once_with("audio2subtitle.mlx")
        mock_provider.generate.assert_called_once()
        assert str(result) == "/tmp/out.srt"

    @patch("audio2subtitle.providers.registry")
    def test_custom_provider(self, mock_registry):
        from audio2subtitle.client import generate

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.output_path = "/tmp/out.srt"
        mock_provider.generate.return_value = mock_response
        mock_registry.get.return_value = mock_provider

        generate("/tmp/in.wav", "/tmp/out.srt", provider_name="custom.provider")

        mock_registry.get.assert_called_once_with("custom.provider")

    @patch("audio2subtitle.providers.registry")
    def test_passes_language(self, mock_registry):
        from audio2subtitle.client import generate

        mock_provider = _RecordingProvider()
        mock_registry.get.return_value = mock_provider

        generate("/tmp/in.wav", "/tmp/out.srt", language="en")

        assert mock_provider.last_request.language == "en"
