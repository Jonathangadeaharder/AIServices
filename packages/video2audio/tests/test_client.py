from unittest.mock import MagicMock, patch


class TestExtract:
    @patch("video2audio.providers.registry")
    def test_calls_provider(self, mock_registry):
        from video2audio.client import extract

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.output_path = "/tmp/out.wav"
        mock_provider.generate.return_value = mock_response
        mock_registry.get.return_value = mock_provider

        result = extract("/tmp/video.mp4", "/tmp/out.wav")

        mock_registry.get.assert_called_once_with("video2audio.ffmpeg")
        mock_provider.generate.assert_called_once()
        assert str(result) == "/tmp/out.wav"

    @patch("video2audio.providers.registry")
    def test_custom_provider(self, mock_registry):
        from video2audio.client import extract

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.output_path = "/tmp/out.wav"
        mock_provider.generate.return_value = mock_response
        mock_registry.get.return_value = mock_provider

        extract("/tmp/video.mp4", "/tmp/out.wav", provider_name="custom.provider")

        mock_registry.get.assert_called_once_with("custom.provider")

    @patch("video2audio.providers.registry")
    def test_passes_format_and_sample_rate(self, mock_registry):
        from video2audio.client import extract

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.output_path = "/tmp/out.mp3"
        mock_provider.generate.return_value = mock_response
        mock_registry.get.return_value = mock_provider

        extract("/tmp/video.mp4", "/tmp/out.mp3", output_format="mp3", sample_rate=22050)

        call_args = mock_provider.generate.call_args
        req = call_args[0][0]
        assert req.output_format == "mp3"
        assert req.sample_rate == 22050
