from unittest.mock import MagicMock, patch

from text2image.client import _div8


class TestDiv8:
    def test_rounds_down(self):
        assert _div8(513) == 512

    def test_rounds_non_multiple(self):
        assert _div8(1001) == 1000

    def test_minimum_512(self):
        assert _div8(100) == 512

    def test_large_value(self):
        assert _div8(1920) == 1920

    def test_already_divisible(self):
        assert _div8(1000) == 1000


class TestGenerate:
    @patch("text2image.providers.registry")
    def test_calls_provider(self, mock_registry):
        from text2image.client import generate

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.output_path = "/tmp/out.png"
        mock_provider.generate.return_value = mock_response
        mock_registry.get.return_value = mock_provider

        result = generate("test prompt", "/tmp/out.png")

        mock_registry.get.assert_called_once_with("text2image.mlx")
        mock_provider.generate.assert_called_once()
        assert str(result) == "/tmp/out.png"

    @patch("text2image.providers.registry")
    def test_custom_provider(self, mock_registry):
        from text2image.client import generate

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.output_path = "/tmp/out.png"
        mock_provider.generate.return_value = mock_response
        mock_registry.get.return_value = mock_provider

        generate("test", "/tmp/out.png", provider_name="custom.provider")

        mock_registry.get.assert_called_once_with("custom.provider")

    @patch("text2image.providers.registry")
    def test_width_height_rounded_to_div8(self, mock_registry):
        from text2image.client import generate

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.output_path = "/tmp/out.png"
        mock_provider.generate.return_value = mock_response
        mock_registry.get.return_value = mock_provider

        generate("test", "/tmp/out.png", width=1001, height=721)

        call_args = mock_provider.generate.call_args
        req = call_args[0][0]
        assert req.width == 1000
        assert req.height == 720
