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
    @patch("image2image.providers.registry")
    def test_calls_provider(self, mock_registry):
        from image2image.client import generate

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.output_path = "/tmp/out.png"
        mock_provider.generate.return_value = mock_response
        mock_registry.get.return_value = mock_provider

        result = generate("/tmp/input.png", "/tmp/out.png", prompt="test")

        mock_registry.get.assert_called_once_with("image2image.mlx")
        mock_provider.generate.assert_called_once()
        assert str(result) == "/tmp/out.png"

    @patch("image2image.providers.registry")
    def test_custom_provider(self, mock_registry):
        from image2image.client import generate

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.output_path = "/tmp/out.png"
        mock_provider.generate.return_value = mock_response
        mock_registry.get.return_value = mock_provider

        generate("/tmp/input.png", "/tmp/out.png", prompt="test", provider_name="custom.provider")

        mock_registry.get.assert_called_once_with("custom.provider")

    @patch("image2image.providers.registry")
    def test_passes_strength(self, mock_registry):
        from image2image.client import generate

        mock_provider = _RecordingProvider()
        mock_registry.get.return_value = mock_provider

        generate("/tmp/input.png", "/tmp/out.png", prompt="test", strength=0.8)

        assert mock_provider.last_request.strength == 0.8
