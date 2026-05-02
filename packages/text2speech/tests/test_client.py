from unittest.mock import MagicMock, patch


class TestGenerate:
    @patch("text2speech.providers.registry")
    def test_calls_provider(self, mock_registry):
        from text2speech.client import generate

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.output_path = "/tmp/out.wav"
        mock_provider.generate.return_value = mock_response
        mock_registry.get.return_value = mock_provider

        result = generate("Hello world", "/tmp/out.wav")

        mock_registry.get.assert_called_once_with("text2speech.fish_mlx")
        mock_provider.generate.assert_called_once()
        assert str(result) == "/tmp/out.wav"

    @patch("text2speech.providers.registry")
    def test_custom_provider(self, mock_registry):
        from text2speech.client import generate

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.output_path = "/tmp/out.wav"
        mock_provider.generate.return_value = mock_response
        mock_registry.get.return_value = mock_provider

        generate("Hello", "/tmp/out.wav", provider_name="custom.provider")

        mock_registry.get.assert_called_once_with("custom.provider")
        mock_provider.generate.assert_called_once()

    @patch("text2speech.providers.registry")
    def test_passes_voice_params(self, mock_registry):
        from text2speech.client import generate

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.output_path = "/tmp/out.wav"
        mock_provider.generate.return_value = mock_response
        mock_registry.get.return_value = mock_provider

        generate("Hello", "/tmp/out.wav", emotion="happy", voice_id="v1")

        mock_provider.generate.assert_called_once()
        call_args = mock_provider.generate.call_args
        req = call_args[0][0]
        assert req.emotion == "happy"
        assert req.voice_id == "v1"
