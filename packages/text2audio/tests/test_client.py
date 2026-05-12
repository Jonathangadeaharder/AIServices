from unittest.mock import MagicMock, patch


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

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.output_path = "/tmp/out.wav"
        mock_provider.generate.return_value = mock_response
        mock_registry.get.return_value = mock_provider

        generate("rain sounds", "/tmp/out.wav", category="ambient", duration_seconds=30.0)

        call_args = mock_provider.generate.call_args
        req = call_args[0][0]
        assert req.category.value == "ambient"
        assert req.duration_seconds == 30.0
