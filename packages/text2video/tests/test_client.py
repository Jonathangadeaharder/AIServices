from unittest.mock import MagicMock, patch

from text2video.client import _div8, _frames_for_seconds


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

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.output_path = "/tmp/out.mp4"
        mock_provider.generate.return_value = mock_response
        mock_registry.get.return_value = mock_provider

        generate("test", "/tmp/out.mp4", width=705, height=481)

        call_args = mock_provider.generate.call_args
        req = call_args[0][0]
        assert req.width == 704
        assert req.height == 480
