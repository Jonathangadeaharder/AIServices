from unittest.mock import MagicMock, patch


class TestGenerate:
    @patch("image2image.providers.registry")
    def test_calls_provider(self, mock_registry):
        from image2image.client import generate

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.output_path = "/tmp/out.png"
        mock_provider.generate.return_value = mock_response
        mock_registry.get.return_value = mock_provider

        result = generate("/tmp/in.png", "test prompt", "/tmp/out.png")

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

        generate("/tmp/in.png", "test", "/tmp/out.png", provider_name="custom.provider")

        mock_registry.get.assert_called_once_with("custom.provider")

    @patch("image2image.providers.registry")
    def test_passes_strength(self, mock_registry):
        from image2image.client import generate

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.output_path = "/tmp/out.png"
        mock_provider.generate.return_value = mock_response
        mock_registry.get.return_value = mock_provider

        generate("/tmp/in.png", "test", "/tmp/out.png", strength=0.8)

        call_args = mock_provider.generate.call_args
        req = call_args[0][0]
        assert req.strength == 0.8
