from unittest.mock import MagicMock, patch

from text2video.client import _div8, _frames_for_seconds

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

class TestDiv8:
    def test_minimum_64(self):
        assert _div8(10) == 64

    def test_rounds_down(self):
        assert _div8(65) == 64

    def test_already_divisible(self):
        assert _div8(640) == 640

class TestFramesForSeconds:
    def test_4s_24fps(self):
        assert _frames_for_seconds(4, 24) == 97

    def test_1s_24fps(self):
        assert _frames_for_seconds(1, 24) == 25

    def test_2s_24fps(self):
        assert _frames_for_seconds(2, 24) == 49

class TestGenerate:
    @patch("text2video.providers.registry")
    def test_calls_provider(self, mock_registry):
        from text2video.client import generate

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.output_path = "/tmp/out.mp4"
        mock_provider.generate.return_value = mock_response
        mock_registry.get.return_value = mock_provider

        result = generate("test prompt", "/tmp/out.mp4")

        mock_registry.get.assert_called_once_with("text2video.mlx")
        mock_provider.generate.assert_called_once()
        assert str(result) == "/tmp/out.mp4"

    @patch("text2video.providers.registry")
    def test_custom_provider(self, mock_registry):
        from text2video.client import generate

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.output_path = "/tmp/out.mp4"
        mock_provider.generate.return_value = mock_response
        mock_registry.get.return_value = mock_provider

        generate("test", "/tmp/out.mp4", provider_name="custom.provider")

        mock_registry.get.assert_called_once_with("custom.provider")

    @patch("text2video.providers.registry")
    def test_width_height_rounded_to_div8(self, mock_registry):
        from text2video.client import generate

        mock_provider = _RecordingProvider()
        mock_registry.get.return_value = mock_provider

        generate("test", "/tmp/out.mp4", width=705, height=481)

        assert mock_provider.last_request.width == 704
        assert mock_provider.last_request.height == 480
