from unittest.mock import MagicMock, patch


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

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.output_path = "/tmp/out.srt"
        mock_provider.generate.return_value = mock_response
        mock_registry.get.return_value = mock_provider

        generate("/tmp/in.wav", "/tmp/out.srt", language="en")

        call_args = mock_provider.generate.call_args
        req = call_args[0][0]
        assert req.language == "en"
