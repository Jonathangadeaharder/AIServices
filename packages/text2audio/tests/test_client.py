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
        self.entries = []
        self.language = None

class TestGenerate:
    @patch("text2audio.providers.registry")
    def test_calls_provider(self, mock_registry):
        from text2audio.client import generate

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.output_path = "/tmp/out.wav"
        mock_provider.generate.return_value = mock_response
        mock_registry.get.return_value = mock_provider

        result = generate("calm piano", "/tmp/out.wav")

        mock_registry.get.assert_called_once_with("text2audio.mlx")
        mock_provider.generate.assert_called_once()
        assert str(result) == "/tmp/out.wav"

    @patch("text2audio.providers.registry")
    def test_custom_provider(self, mock_registry):
        from text2audio.client import generate

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.output_path = "/tmp/out.wav"
        mock_provider.generate.return_value = mock_response
        mock_registry.get.return_value = mock_provider

        generate("test", "/tmp/out.wav", provider_name="custom.provider")

        mock_registry.get.assert_called_once_with("custom.provider")

    @patch("text2audio.providers.registry")
    def test_passes_category_and_duration(self, mock_registry):
        from text2audio.client import generate

        mock_provider = _RecordingProvider()
        mock_registry.get.return_value = mock_provider

        generate("rain sounds", "/tmp/out.wav", category="ambient", duration_seconds=30.0)

        assert mock_provider.last_request.category.value == "ambient"
        assert mock_provider.last_request.duration_seconds == 30.0
